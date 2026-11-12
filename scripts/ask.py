"""Asks an agent a question and prints the answer with the offers it picked.

Run from the project root:
    uv run python scripts/ask.py "Which remote offers include a training budget?"
    uv run python scripts/ask.py --agent graph "Is the average salary higher in Krakow or Warsaw?"
"""

import argparse

from offers.config import Settings
from offers.graph_agent import ask_graph
from offers.io import load_offers
from offers.llm import create_client
from offers.pyai_agent import ask_agent
from offers.store import connect
from offers.tools import AgentDeps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask an agent about the job offers.")
    parser.add_argument("question")
    parser.add_argument("--agent", choices=["pyai", "graph"], default="pyai")
    args = parser.parse_args()

    settings = Settings()
    with connect(settings.database_url) as conn:
        deps = AgentDeps(
            client=create_client(settings),
            conn=conn,
            settings=settings,
            offers=load_offers("offers.json"),
        )
        ask = ask_agent if args.agent == "pyai" else ask_graph
        output = ask(deps, args.question)

    print(output.answer)
    if output.offers:
        print(f"\nOffers: {', '.join(output.offers)}")
