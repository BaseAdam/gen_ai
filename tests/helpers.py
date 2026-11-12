"""Fakes and builders shared by several test modules."""

import json
from typing import Any

from google.genai import types
from pydantic import SecretStr

from offers.config import Settings
from offers.embeddings import EMBEDDING_DIMENSIONS
from offers.models import Offer
from offers.tools import AgentDeps, SearchHit


def text_response(text: str) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def calls_response(*calls: tuple[str, dict[str, Any]]) -> types.GenerateContentResponse:
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in calls
    ]
    content = types.Content(role="model", parts=parts)
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def call_response(name: str, **args: Any) -> types.GenerateContentResponse:
    return calls_response((name, args))


class ScriptedModels:
    """Returns the scripted responses in order and records every request."""

    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append({**kwargs, "contents": list(kwargs["contents"])})
        return self.responses.pop(0)

    @property
    def sent(self) -> list[list[types.Content]]:
        return [call["contents"] for call in self.calls]


class ScriptedClient:
    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self.models = ScriptedModels(responses)


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (EMBEDDING_DIMENSIONS - len(head))


def relevance(*scores: int) -> str:
    items = [{"index": number, "score": score} for number, score in enumerate(scores, start=1)]
    return json.dumps({"scores": items})


def hit(title: str, score: float) -> SearchHit:
    return SearchHit(
        title=title,
        company="X",
        city="Krakow",
        remote=True,
        salary_min=None,
        salary_max=None,
        fragment="...",
        relevance=score,
    )


def make_settings(**overrides: Any) -> Settings:
    return Settings(
        gemini_api_key=SecretStr("test-key"), gemini_model="chat-model", _env_file=None, **overrides
    )


def make_deps(
    offers: list[Offer], client: Any = None, conn: Any = None, **settings: Any
) -> AgentDeps:
    return AgentDeps(client=client, conn=conn, settings=make_settings(**settings), offers=offers)
