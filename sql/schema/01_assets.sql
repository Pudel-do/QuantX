CREATE TABLE IF NOT EXISTS assets (
    asset_id INTEGER NOT NULL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    long_name TEXT,
    short_name TEXT,
    currency TEXT,
    active_flag BOOLEAN NOT NULL DEFAULT true
);