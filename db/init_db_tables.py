from pathlib import Path
from sqlalchemy import text
from engine import get_engine

engine = get_engine()

sql_dirs = [
    "sql/schema"
]

with engine.begin() as conn:
    for d in sql_dirs:
        for sql_file in sorted(Path(d).glob("*.sql")):
            sql = sql_file.read_text()
            conn.execute(text(sql))