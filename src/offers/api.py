from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from pydantic import BaseModel, Field

from offers.analysis import filter_offers
from offers.assistant import answer_question
from offers.config import Settings
from offers.io import load_offers
from offers.llm import convert_offer_text_to_structure, create_client, stream_interview_tips
from offers.models import Offer

app = FastAPI(title="Offers API")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_client() -> genai.Client:
    return create_client(get_settings())


def get_offers() -> list[Offer]:
    return load_offers("offers.json")


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClientDep = Annotated[genai.Client, Depends(get_client)]
OffersDep = Annotated[list[Offer], Depends(get_offers)]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


class AskResponse(BaseModel):
    answer: str


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
