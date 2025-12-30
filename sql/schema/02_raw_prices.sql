CREATE TABLE IF NOT EXISTS raw_prices (
    ticker_id NOT NULL,
    date DATE NOT NULL,
    high REAL,
    low REAL,
    open REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker_id, date),
    FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id)
)