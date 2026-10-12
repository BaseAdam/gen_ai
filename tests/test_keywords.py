from offers.keywords import bm25_scores, tokenize


def test_tokenize_lowercases_drops_short_words_and_cuts_endings() -> None:
    assert tokenize("Company provides modern equipment in office") == [
        "compan",
        "provid",
        "modern",
        "equipm",
        "office",
    ]


def test_tokenize_skips_stop_words() -> None:
    assert tokenize("Where is the equipment, which they provide?") == ["equipm", "provid"]


def test_bm25_scores_only_documents_with_query_words() -> None:
    scores = bm25_scores("equipment", ["new equipment", "new office", "good coffee"])

    assert scores[0] > 0
    assert scores[1] == scores[2] == 0


def test_bm25_rare_word_outweighs_common_word() -> None:
    documents = ["work office", "work office", "work office", "equipment office"]

    scores = bm25_scores("work equipment", documents)

    assert scores[3] > scores[0]


def test_bm25_matches_inflected_forms() -> None:
    scores = bm25_scores("laptops", ["new laptop", "new office"])

    assert scores[0] > 0


def test_bm25_without_documents() -> None:
    assert bm25_scores("equipment", []) == []


def test_bm25_ignores_stop_words() -> None:
    scores = bm25_scores(
        "Where will I get a laptop?", ["the office where we work", "new equipment"]
    )

    assert scores == [0.0, 0.0]
