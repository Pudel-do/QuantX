from contextlib import contextmanager
from sqlalchemy.engine import Connection
from db.engine import get_engine


@contextmanager
def get_connection():
    engine = get_engine()
    conn = engine.connect()

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
