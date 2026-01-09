CREATE TABLE IF NOT EXISTS price_features (
    asset_id INTEGER NOT NULL,
    date DATE NOT NULL,
    price REAL,
    sma_short REAL,
    sma_long REAL,
    ema_short REAL,
    ema_long REAL,
    rsi REAL,
    macd REAL,
    atr REAL,
    obv INTEGER,
    PRIMARY KEY (asset_id, date),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);