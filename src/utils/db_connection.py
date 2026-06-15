# src/utils/db_connection.py

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# .env file load karo
load_dotenv()

def get_engine():
    """Database connection engine return karta hai."""
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
    )
    engine = create_engine(db_url, echo=False)
    return engine


def test_connection():
    """Simple connection test."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connected! PostgreSQL version: {version}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    test_connection()