from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine
from db.engine import get_engine
from core.finance_adapter import FinanceAdapter
from misc.schema_utils import *
from misc.utils import read_json

class AssetRepository:

    def __init__(self):
        self.engine = get_engine()

    # -------------------------
    # Public API
    # -------------------------

    def add_asset(self, ticker: str):

        ticker = ticker.upper().strip()
        if ticker not in self.get_tickers():
            infos = FinanceAdapter(ticker).get_stock_infos()
            metadata = pd.DataFrame(
                {
                "ticker": ticker,
                "long_name": infos.get("longName"),
                "short_name": infos.get("shortName"),
                "currency": infos.get("currency")
                },
                index=[0]
            )

            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT OR IGNORE INTO assets (ticker, long_name, short_name, currency)
                        VALUES (:ticker, :long_name, :short_name, :currency)
                        """
                    ),
                    metadata.to_dict(orient="records")
            )
                
        else:
            pass

        return None

    def update_active_flag(self, ticker_list):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE assets
                    SET active_flag = false
                    """
                )
            )

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE assets
                    SET active_flag = true
                    WHERE ticker IN :ticker_list
                    """
                ).bindparams(bindparam("ticker_list", expanding=True)),
                {"ticker_list": ticker_list}
            )
        
        return None

    def get_active_ids(self):
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT asset_id
                    FROM assets
                    WHERE active_flag = true
                    ORDER BY asset_id
                    """)).fetchall()
            
        return [x[0] for x in rows]

    def get_ids(self):
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT asset_id
                    FROM assets
                    ORDER BY asset_id
                    """)).fetchall()
            
        return [x[0] for x in rows]

    def get_tickers(self):
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT ticker
                    FROM assets
                    ORDER BY ticker
                    """)).fetchall()
        return [x[0] for x in rows]
