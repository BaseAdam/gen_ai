from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

import httpx
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import errors
from pydantic import BaseModel, Field

from offers.analysis import filter_offers
from offers.assistant import answer_question
from offers.config import Settings
from offers.io import load_offers
from offers.llm import convert_offer_text_to_structure, create_client, stream_interview_tips
from offers.models import Offer, RagAnswer
from offers.rag import answer_with_rag
from offers.store import Connection, connect

app = FastAPI(title="Offers API")


@app.exception_handler(httpx.TimeoutException)
def handle_timeout(request: Request, exc: httpx.TimeoutException) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": "Model nie odpowiedział na czas"})


@app.exception_handler(errors.APIError)
def handle_api_error(request: Request, exc: errors.APIError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "Błąd usługi modelu"})


@app.exception_handler(psycopg.OperationalError)
def handle_database_error(request: Request, exc: psycopg.OperationalError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Baza danych jest niedostępna"})


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_client() -> genai.Client:
    return create_client(get_settings())


def get_offers() -> list[Offer]:
    return load_offers("offers.json")


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_conn(settings: SettingsDep) -> Iterator[Connection]:
    with connect(settings.database_url) as conn:
        yield conn


ConnDep = Annotated[Connection, Depends(get_conn)]
ClientDep = Annotated[genai.Client, Depends(get_client)]
OffersDep = Annotated[list[Offer], Depends(get_offers)]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


class AskResponse(BaseModel):
    answer: str


class RagRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)


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
        raise HTTPException(status_code=404, detail="Nie ma takiej oferty")
    stream = stream_interview_tips(client, offers[index], settings.gemini_model)
    return StreamingResponse(stream, media_type="text/plain; charset=utf-8")


@app.post("/rag")
def rag(request: RagRequest, client: ClientDep, settings: SettingsDep, conn: ConnDep) -> RagAnswer:
    return answer_with_rag(
        client,
        conn,
        request.question,
        settings.gemini_model,
        settings.gemini_embedding_model,
        request.top_k,
    )
