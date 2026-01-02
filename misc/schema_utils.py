import pandas as pd
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.engine import Engine
from db.engine import get_engine
import logging

log = logging.getLogger(__name__)


def drop_all_tables() -> None:
    """
    Löscht ALLE Tabellen!
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        tables = conn.execute(text("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%';
        """)).fetchall()

        for (table_name,) in tables:
            log.info(f"Dropping table: {table_name}")
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

        conn.execute(text("PRAGMA foreign_keys = ON"))


def create_schema_from_sql(
    schema_dir: str | Path,
) -> None:
    """
    Erstellt alle Tabellen aus SQL-Dateien
    """
    schema_dir = Path(schema_dir)
    engine = get_engine()

    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema dir not found: {schema_dir}")

    with engine.begin() as conn:
        for sql_file in sorted(schema_dir.glob("*.sql")):
            log.info(f"Creating table from {sql_file.name}")
            sql = sql_file.read_text()
            conn.execute(text(sql))


def init_schema(
    sql_dir: list | Path,
    reset: bool = False,
) -> None:
    """
    Initialisiert das Schema.
    reset=True → DROP ALL + CREATE
    """
    if reset:
        drop_all_tables()

    for dir in sql_dir:
        create_schema_from_sql(dir)


def get_ticker_id(df: pd.DataFrame) -> pd.DataFrame:
    
    engine = get_engine()
    ticker_map = pd.read_sql(
        "SELECT ticker_id, ticker FROM tickers",
        engine
    ).set_index("ticker")["ticker_id"]
    df["ticker_id"] = df["ticker"].map(ticker_map)
    return df

def write_sql(sql: str, data) -> None:
    
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(sql),
            data.to_dict(orient="records")
        )

def read_sql(sql, params):
    engine = get_engine()

    if params is None:
        df = pd.read_sql(
            sql=sql,
            con=engine
        )
    else:
        df = pd.read_sql(
            sql=sql,
            con=engine,
            params=params
        )
    return df
