from google import genai
from google.genai import types

from offers.models import Offer

EMBEDDING_DIMENSIONS = 768


def offer_to_text(offer: Offer) -> str:
    mode = "remote" if offer.remote else "on-site"
    skills = ", ".join(offer.skills) or "not specified"
    return f"{offer.title} at {offer.company}, {offer.city}, {mode}. Requirements: {skills}."


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
