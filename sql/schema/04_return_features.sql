CREATE TABLE IF NOT EXISTS return_features(
    asset_id INTEGER NOT NULL,
    date DATE NOT NULL, 
    return REAL,
    return_vola REAL,
    PRIMARY KEY (asset_id, date),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

