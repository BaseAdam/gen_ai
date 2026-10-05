from offers.config import Settings
from offers.store import connect, create_schema

if __name__ == "__main__":
    settings = Settings()

    with connect(settings.database_url) as conn:
        create_schema(conn)
        row = conn.execute("SELECT count(*) FROM offers").fetchone()
        print("Tabela offers gotowa, wierszy:", row[0] if row else 0)
