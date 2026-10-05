import math

from google.genai import types

from offers.config import Settings
from offers.llm import create_client

EMBEDDING_MODEL = "gemini-embedding-001"

TEXTS = [
    "Szukamy programisty Python do pracy z FastAPI",
    "Poszukujemy backend developera znającego Pythona i REST API",
    "Zatrudnimy kucharza do restauracji włoskiej",
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    settings = Settings()
    client = create_client(settings)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=TEXTS,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=768),
    )

    vectors = [embedding.values or [] for embedding in response.embeddings or []]

    print(f"{len(vectors)} wektory, każdy ma {len(vectors[0])} liczb")
    print("początek pierwszego:", vectors[0][:5])
    print(f"python vs backend: {cosine_similarity(vectors[0], vectors[1]):.3f}")
    print(f"python vs kucharz: {cosine_similarity(vectors[0], vectors[2]):.3f}")
