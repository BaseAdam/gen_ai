import math

from google import genai
from google.genai import types

from offers.models import Offer, ScoredOffer

EMBEDDING_DIMENSIONS = 768


def offer_to_text(offer: Offer) -> str:
    mode = "zdalnie" if offer.remote else "stacjonarnie"
    skills = ", ".join(offer.skills) or "brak danych"
    return f"{offer.title} w {offer.company}, {offer.city}, {mode}. Wymagania: {skills}."


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def embed_texts(
    client: genai.Client, texts: list[str], model: str, task_type: str
) -> list[list[float]]:
    config = types.EmbedContentConfig(
        task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS
    )
    response = client.models.embed_content(model=model, contents=texts, config=config)
    vectors = [embedding.values or [] for embedding in response.embeddings or []]
    if len(vectors) != len(texts) or not all(vectors):
        raise ValueError("Incomplete embedding response")
    return vectors


def embed_groups(
    client: genai.Client, groups: list[list[str]], model: str, task_type: str
) -> list[list[list[float]]]:
    texts = [text for group in groups for text in group]
    vectors = embed_texts(client, texts, model, task_type)

    result = []
    position = 0
    for group in groups:
        result.append(vectors[position : position + len(group)])
        position += len(group)
    return result


def rank_offers(
    query_vector: list[float],
    offer_vectors: list[list[float]],
    offers: list[Offer],
    top_k: int = 5,
) -> list[ScoredOffer]:
    scored = [
        ScoredOffer(offer=offer, score=cosine_similarity(query_vector, vector))
        for vector, offer in zip(offer_vectors, offers, strict=True)
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]
