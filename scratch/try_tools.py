from offers.config import Settings
from offers.io import load_offers
from offers.llm import create_client
from offers.observability import configure_logging
from offers.store import connect
from offers.tools import AgentDeps, list_offers, salary_stats, search_job_ads

if __name__ == "__main__":
    configure_logging()
    settings = Settings()
    with connect(settings.database_url) as conn:
        deps = AgentDeps(
            client=create_client(settings),
            conn=conn,
            settings=settings,
            offers=load_offers("offers.json"),
        )

        print("list_offers(city='Krakow', remote=True)")
        for row in list_offers(deps, city="Krakow", remote=True):
            print(f"  {row.model_dump_json()}")

        print("\nsearch_job_ads('flexible working hours')")
        hits = search_job_ads(deps, "flexible working hours")
        for hit in hits:
            print(
                f"  {hit.title} ({hit.company}, {hit.city}): {hit.fragment.splitlines()[-1][:90]}"
            )

        titles = [hit.title for hit in hits]
        print(f"\nsalary_stats(titles={titles})")
        print(f"  {salary_stats(deps, titles=titles).model_dump_json()}")
