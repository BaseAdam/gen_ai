import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import httpx
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import errors
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from pydantic_ai import AgentRunError, ModelAPIError, ModelHTTPError, ModelTimeoutError
from pydantic_ai.models import Model

from offers.analysis import filter_offers
from offers.assistant import answer_question
from offers.config import Settings
from offers.io import load_offers
from offers.llm import convert_offer_text_to_structure, create_client, stream_interview_tips
from offers.models import Offer, RagAnswer
from offers.observability import configure_logging, current_request, elapsed_ms, request_context
from offers.pyai_agent import ask_agent
from offers.rag import answer_with_rag
from offers.store import Connection
from offers.tools import AgentDeps

logger = logging.getLogger(__name__)

POOL_MAX_SIZE = 10
POOL_TIMEOUT_SECONDS = 5.0
FRONTEND_DIR = Path("frontend/dist")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json_format=settings.log_json)
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=POOL_MAX_SIZE,
        timeout=POOL_TIMEOUT_SECONDS,
        check=ConnectionPool.check_connection,
        open=False,
    )
    with pool:
        app.state.pool = pool
        yield


app = FastAPI(title="Offers API", lifespan=lifespan)


def serve_frontend(app: FastAPI, directory: Path) -> None:
    """Serves the built React app if it exists (after npm run build, and in the Docker image).

    Frontend routes have low priority: FastAPI matches the API paths first.
    """
    if directory.is_dir():
        app.frontend("/", directory=directory)


serve_frontend(app, FRONTEND_DIR)


@app.middleware("http")
async def track_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    with request_context(request_id) as context:
        response = await call_next(request)
        duration = elapsed_ms(start)
        logger.info(
            "%s %s -> %d in %.0f ms, tokens: input=%d output=%d thinking=%d",
            request.method,
            request.url.path,
            response.status_code,
            duration,
            context.input_tokens,
            context.output_tokens,
            context.thinking_tokens,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration),
                "input_tokens": context.input_tokens,
                "output_tokens": context.output_tokens,
                "thinking_tokens": context.thinking_tokens,
            },
        )
    response.headers["X-Request-ID"] = request_id
    return response


def model_error(timed_out: bool) -> JSONResponse:
    if timed_out:
        return JSONResponse(
            status_code=504, content={"detail": "The model did not respond in time"}
        )
    return JSONResponse(status_code=502, content={"detail": "Model service error"})


@app.exception_handler(httpx.TransportError)
def handle_transport_error(request: Request, exc: httpx.TransportError) -> JSONResponse:
    return model_error(timed_out=isinstance(exc, httpx.TimeoutException))


@app.exception_handler(errors.APIError)
def handle_api_error(request: Request, exc: errors.APIError) -> JSONResponse:
    return model_error(timed_out=exc.code == 504)


@app.exception_handler(ModelAPIError)
def handle_agent_model_error(request: Request, exc: ModelAPIError) -> JSONResponse:
    deadline = isinstance(exc, ModelHTTPError) and exc.status_code == 504
    return model_error(timed_out=deadline or isinstance(exc, ModelTimeoutError))


@app.exception_handler(AgentRunError)
def handle_agent_error(request: Request, exc: AgentRunError) -> JSONResponse:
    return JSONResponse(
        status_code=502, content={"detail": "The agent could not finish the answer"}
    )


@app.exception_handler(psycopg.OperationalError)
def handle_database_error(request: Request, exc: psycopg.OperationalError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Database is unavailable"})


@lru_cache
def get_client() -> genai.Client:
    return create_client(get_settings())


def get_offers() -> list[Offer]:
    return load_offers("offers.json")


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_conn(request: Request) -> Iterator[Connection]:
    pool: ConnectionPool[Connection] = request.app.state.pool
    with pool.connection() as conn:
        yield conn


ConnDep = Annotated[Connection, Depends(get_conn)]
ClientDep = Annotated[genai.Client, Depends(get_client)]
OffersDep = Annotated[list[Offer], Depends(get_offers)]


def get_agent_deps(
    client: ClientDep, conn: ConnDep, settings: SettingsDep, offers: OffersDep
) -> AgentDeps:
    return AgentDeps(client=client, conn=conn, settings=settings, offers=offers)


def get_agent_model() -> Model | None:
    """None means Gemini from the settings; tests put a scripted model here."""
    return None


AgentDepsDep = Annotated[AgentDeps, Depends(get_agent_deps)]
AgentModelDep = Annotated[Model | None, Depends(get_agent_model)]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


class AskResponse(BaseModel):
    answer: str


class AgentUsage(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int


class AgentResponse(BaseModel):
    answer: str
    offers: list[Offer]
    steps: list[list[str]] = Field(
        description="Tools the agent called in each step; the last step writes the answer"
    )
    usage: AgentUsage


class RagRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)
    city: str | None = None
    remote: bool | None = None


@app.get("/offers")
def list_offers(
    offers: OffersDep,
    city: str | None = None,
    remote: bool | None = None,
    skill: str | None = None,
) -> list[Offer]:
    return filter_offers(offers, city=city, remote=remote, skill=skill)


@app.post("/offers/extract")
def extract_offer(request: ExtractRequest, client: ClientDep, settings: SettingsDep) -> Offer:
    return convert_offer_text_to_structure(client, request.text, settings.gemini_model)


@app.post("/ask")
def ask(
    request: AskRequest, client: ClientDep, settings: SettingsDep, offers: OffersDep
) -> AskResponse:
    answer = answer_question(client, request.question, offers, settings.gemini_model)
    return AskResponse(answer=answer)


@app.get("/offers/{index}/tips")
def offer_tips(
    index: int, client: ClientDep, settings: SettingsDep, offers: OffersDep
) -> StreamingResponse:
    if not 0 <= index < len(offers):
        raise HTTPException(status_code=404, detail="Offer not found")
    stream = stream_interview_tips(client, offers[index], settings.gemini_model)
    return StreamingResponse(stream, media_type="text/plain; charset=utf-8")


@app.post("/rag")
def rag(request: RagRequest, client: ClientDep, settings: SettingsDep, conn: ConnDep) -> RagAnswer:
    return answer_with_rag(
        client,
        conn,
        request.question,
        chat_model=settings.gemini_model,
        embedding_model=settings.gemini_embedding_model,
        top_k=request.top_k,
        city=request.city,
        remote=request.remote,
        min_score=settings.rag_min_score,
        use_rerank=settings.rag_rerank,
    )


def pick_offers(offers: list[Offer], titles: list[str]) -> list[Offer]:
    by_title = {offer.title: offer for offer in offers}
    unknown = [title for title in titles if title not in by_title]
    if unknown:
        logger.warning("The agent named offers that do not exist: %s", unknown)
    return [by_title[title] for title in dict.fromkeys(titles) if title in by_title]


@app.post("/agent")
def agent(request: AskRequest, deps: AgentDepsDep, model: AgentModelDep) -> AgentResponse:
    output = ask_agent(deps, request.question, model=model)
    context = current_request.get()
    if context is None:
        raise RuntimeError("The agent ran outside the request middleware")
    return AgentResponse(
        answer=output.answer,
        offers=pick_offers(deps.offers, output.offers),
        steps=context.steps,
        usage=AgentUsage(
            llm_calls=context.llm_calls,
            input_tokens=context.input_tokens,
            output_tokens=context.output_tokens,
            thinking_tokens=context.thinking_tokens,
        ),
    )
