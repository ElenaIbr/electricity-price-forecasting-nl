from sqlalchemy import text
from src.db.connection import get_engine


def main() -> None:
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print(result.fetchone()[0])


if __name__ == "__main__":
    main()
