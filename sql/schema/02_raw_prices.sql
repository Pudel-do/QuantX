CREATE TABLE IF NOT EXISTS raw_prices (
    asset_id INTEGER NOT NULL,
    date DATE NOT NULL,
    high REAL,
    low REAL,
    open REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    PRIMARY KEY (asset_id, date),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
)