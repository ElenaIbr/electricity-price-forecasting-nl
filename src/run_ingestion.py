# ================================
# RAW data ingestion orchestrator
# Diploma project
#
# Runs all ingestion scripts in a fixed, reproducible order.
# ================================

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# ---- Import ingestion jobs ----
from ingestion.entsoe_prices import main as entsoe_prices
from ingestion.entsoe_load import main as entsoe_load
from ingestion.entsoe_solar import main as entsoe_solar
from ingestion.entsoe_wind import main as entsoe_wind
from ingestion.weather_open_meteo import main as weather
from ingestion.gas_prices import main as gas_prices


def main() -> None:
    print("RAW DATA INGESTION PIPELINE START")

    entsoe_prices()
    entsoe_load()
    entsoe_solar()
    entsoe_wind()
    weather()
    gas_prices()

    print("RAW DATA INGESTION PIPELINE DONE")


if __name__ == "__main__":
    main()
