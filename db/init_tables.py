from pathlib import Path
from misc.utils import read_json
from misc.schema_utils import *

ENV = read_json("Parameter.json")["env"]
RESET_SCHEMA = ENV == "dev"

def run():
    sql_dir = [
        "sql/schema",
        "sql/dashboard"
    ]
    init_schema(
    sql_dir=sql_dir,
    reset=RESET_SCHEMA,
)

if __name__ == "__main__":
    run()

