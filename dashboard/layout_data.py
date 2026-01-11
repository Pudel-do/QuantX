import pandas as pd
from misc.utils import read_json
from misc.schema_utils import read_sql
from functools import lru_cache

@lru_cache()
def load_active_assets():
    query = """
    SELECT name
    FROM assets
    WHERE active_flag = 1
    AND benchmark_flag = 0
    ORDER BY name
    """

    df = read_sql(
        sql=query,
        params=None
    )

    
    pass

@lru_cache()
def load_slider_values():

    query = """
    SELECT
    MIN(date) as min_date, 
    MAX(date) as max_date
    FROM return_features
    """ 

    df = read_sql(
        query=query,
        params=None
    )

    min_date = pd.to_datetime(df.loc[0, "min_date"])
    max_date = pd.to_datetime(df.loc[0, "max_date"])

    min_ts = int(min_date.timestamp())
    max_ts = int(max_date.timestamp())

    marks = {
        int(pd.Timestamp(year=y, month=1, day=1).timestamp()): str(y)
        for y in range(min_date.year, max_date.year + 1)
    }

    return {
        "min": min_ts,
        "max": max_ts,
        "value": [min_ts, max_ts],
        "step": 24 * 60 * 60,
        "marks": marks
    }




