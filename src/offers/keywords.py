import math
import re
from collections import Counter

STEM_LENGTH = 6
MIN_WORD_LENGTH = 3

STOP_WORDS = frozenset(
    """
    the and for are was were been being has have had does did can could should would will
    what which who whom whose where when why how
    this that these those there here their they them you your our its
    with from into onto about over under than then also just
    not but any all some each other such only very too
    """.split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    return [
        word[:STEM_LENGTH]
        for word in words
        if len(word) >= MIN_WORD_LENGTH and word not in STOP_WORDS
    ]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized = [tokenize(document) for document in documents]
    if not tokenized:
        return []

    count = len(tokenized)
    average_length = (sum(len(tokens) for tokens in tokenized) / count) or 1
    document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
    query_terms = set(tokenize(query))

    scores = []
    for tokens in tokenized:
        frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = frequency[term]
            if tf == 0:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
            length_norm = 1 - b + b * len(tokens) / average_length
            score += idf * tf * (k1 + 1) / (tf + k1 * length_norm)
        scores.append(score)
    return scores
