# src/data/create_schema.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy import text
from src.utils.db_connection import get_engine


def create_schema():
    """
    SQL schema file padhke database mein apply karta hai.
    Idempotent hai — baar baar chalao, same result milega.
    """
    # Schema file ka path
    schema_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'schema.sql'
    )
    schema_path = os.path.abspath(schema_path)

    print(f"📂 Schema file: {schema_path}")

    # File padhna
    with open(schema_path, 'r', encoding='utf-8') as f:
        sql_script = f.read()

    # Database mein execute karna
    engine = get_engine()
    with engine.connect() as conn:
        # Transaction mein run karo
        with conn.begin():
            # Har statement alag execute karo
            statements = [s.strip() for s in sql_script.split(';') if s.strip()]
            for stmt in statements:
                conn.execute(text(stmt))

    print("✅ Schema created successfully!")
    print("Tables created: patients, admissions, diagnoses, lab_results, medications, readmissions")


def verify_tables():
    """Check karo ki sab tables ban gayi hain."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """))
        tables = [row[0] for row in result]

    expected = ['admissions', 'diagnoses', 'lab_results',
                'medications', 'patients', 'readmissions']

    print("\n📋 Tables in database:")
    for table in tables:
        status = "✅" if table in expected else "ℹ️"
        print(f"  {status} {table}")

    missing = set(expected) - set(tables)
    if missing:
        print(f"\n❌ Missing tables: {missing}")
    else:
        print("\n🎉 All 6 tables present!")


if __name__ == "__main__":
    create_schema()
    verify_tables()