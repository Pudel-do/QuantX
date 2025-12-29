from pathlib import Path
from sqlalchemy import text
from engine import get_engine
from misc.utils import read_json
from misc.schema_utils import *

ENGINE = get_engine()
ENV = read_json("Parameter.json")["env"]
RESET_SCHEMA = ENV == "dev"

def run():
    init_schema(
    engine=ENGINE,
    schema_dir="sql/schema",
    reset=RESET_SCHEMA,
)

if __name__ == "__main__":
    run()

