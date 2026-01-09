CREATE TABLE IF NOT EXISTS assets (
    asset_id INTEGER NOT NULL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    name TEXT,
    long_name TEXT,
    short_name TEXT,
    currency TEXT,
    benchmark_flag BOOLEAN NOT NULL DEFAULT false,
    active_flag BOOLEAN NOT NULL DEFAULT true
);