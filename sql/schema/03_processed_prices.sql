CREATE TABLE IF NOT EXISTS processed_prices (
    asset_id INTEGER NOT NULL,
    date DATE NOT NULL,
    price REAL,
    return REAL,
    sma_short REAL,
    sma_long REAL,
    rsi REAL,
    macd REAL,
    atr REAL,
    obv REAL,
    ret_vola REAL,
    PRIMARY KEY (asset_id, date),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
)