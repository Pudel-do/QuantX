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
def load_return_slider_values():

    query = """
    SELECT DISTINCT date
    FROM return_features
    ORDER BY date
    """ 

    df = read_sql(
        query=query,
        params=None
    )

    date_index = pd.DatetimeIndex(df["date"])
    marks, date_range = _load_slider_values(date_index)
    
    return marks, date_range




def _load_slider_values(date_range):

    raw_marks = {
        i: str(date.year) \
            for i, date in enumerate(date_range)
    }
    seen_years = set()
    marks = {}
    for key, value in raw_marks.items():
        if value not in seen_years:
            marks[key] = value
            seen_years.add(value)
    
    return marks, date_range


