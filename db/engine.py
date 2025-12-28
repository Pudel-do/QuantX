import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from pathlib import Path

_ENGINE: Engine | None = None


def get_engine() -> Engine:
    global _ENGINE

    if _ENGINE is None:
        db_type = os.getenv("DB_TYPE")

        if db_type == "sqlite":
            db_path = os.getenv(
                "SQLITE_PATH",
                Path("db/data.db").absolute()
            )
            connection_string = f"sqlite:///{db_path}"

            _ENGINE = create_engine(
                connection_string,
                echo=False,
                future=True,
                connect_args={"check_same_thread": False}
            )

        elif db_type == "postgres":
            user = os.getenv("PG_USER")
            password = os.getenv("PG_PASSWORD")
            host = os.getenv("PG_HOST", "localhost")
            port = os.getenv("PG_PORT", "5432")
            database = os.getenv("PG_DB")

            connection_string = (
                f"postgresql+psycopg2://{user}:{password}"
                f"@{host}:{port}/{database}"
            )

            _ENGINE = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False,
                future=True
            )

        else:
            raise ValueError(f"Unsupported DB_TYPE: {db_type}")

    return _ENGINE
