CREATE TABLE IF NOT EXISTS tickers (
    ticker_id INTEGER PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    long_name TEXT,
    short_name TEXT,
    currency TEXT
);