from offers.keywords import bm25_scores, tokenize


def test_tokenize_lowercases_drops_short_words_and_cuts_endings() -> None:
    assert tokenize("Zapewniamy nowoczesny sprzęt w biurze") == [
        "zapewn",
        "nowocz",
        "sprzęt",
        "biurze",
    ]


def test_bm25_scores_only_documents_with_query_words() -> None:
    scores = bm25_scores("sprzęt", ["mamy sprzęt", "mamy biuro", "dobra kawa"])

    assert scores[0] > 0
    assert scores[1] == scores[2] == 0


def test_bm25_rare_word_outweighs_common_word() -> None:
    documents = ["praca biuro", "praca biuro", "praca biuro", "sprzęt biuro"]

    scores = bm25_scores("praca sprzęt", documents)

    assert scores[3] > scores[0]


def test_bm25_matches_inflected_forms() -> None:
    scores = bm25_scores("sprzętu", ["nowy sprzęt", "nowe biuro"])

    assert scores[0] > 0


def test_bm25_without_documents() -> None:
    assert bm25_scores("sprzęt", []) == []
