import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_engine():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    return create_engine(database_url)
