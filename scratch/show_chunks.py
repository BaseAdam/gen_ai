from offers.chunking import chunk_offer
from offers.io import load_offers

if __name__ == "__main__":
    offers = load_offers("offers.json")

    for chunk in chunk_offer(offers[0]):
        print(chunk)
        print("-" * 40)

    counts = [len(chunk_offer(offer)) for offer in offers]
    print("chunks per offer:", counts)
    print("total:", sum(counts))
