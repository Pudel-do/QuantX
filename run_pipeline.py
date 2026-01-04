import logging
log = logging.getLogger(__name__)

def run_step(name, fn):
    log.info(f"START {name}")
    try:
        fn()
        log.info(f"SUCCESS {name}")
    except Exception:
        log.exception(f"FAILED {name}")
        raise


def main():
    from db import init_tables
    from scripts.ingest import ingest_assets
    from scripts.ingest import ingest_raw_prices
    from scripts.process import process_prices

    run_step("INITIALIZE TABLES", init_tables.run)
    run_step("INGEST ASSETS", ingest_assets.run)
    run_step("INGEST RAW PRICES", ingest_raw_prices.run)
    run_step("PROCESS PRICES", process_prices.run)

if __name__ == "__main__":
    main()