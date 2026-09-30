from offers.analysis import filter_offers
from offers.io import load_offers, save_offers
from offers.report import print_report

if __name__ == "__main__":
    offers = load_offers("offers.json")
    save_offers(filter_offers(offers, remote=True), "remote_offers.json")
    print_report(offers)
