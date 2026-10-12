from offers.chunking import chunk_offer
from offers.io import load_offers
from offers.keywords import bm25_scores

QUESTION = "Which offers provide equipment for remote work?"

if __name__ == "__main__":
    offers = load_offers("offers.json")
    pairs = [(offer, chunk) for offer in offers for chunk in chunk_offer(offer)]
    scores = bm25_scores(QUESTION, [chunk for _, chunk in pairs])

    ranked = sorted(zip(scores, pairs, strict=True), key=lambda item: item[0], reverse=True)
    for score, (offer, chunk) in ranked[:8]:
        section = chunk.splitlines()[1]
        print(f"{score:5.2f}  {offer.title} @ {offer.company}  [{section}]")
