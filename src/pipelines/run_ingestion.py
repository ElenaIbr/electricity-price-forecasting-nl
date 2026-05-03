# ================================
# RAW data ingestion orchestrator
# Diploma project
# ================================

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.yfinance_co2_price import main as co2_price
from src.ingestion.yfinance_gas_price import main as gas_price

from src.ingestion.open_meteo_weather_forecast import main as weather_forecast
from src.ingestion.open_meteo_weather_actual import main as weather_actual

from src.ingestion.entsoe_load_15 import main as load_15
from src.ingestion.entsoe_load_forecast_15 import main as load_forecast_15

from src.ingestion.entsoe_imbalance_price_15 import main as imbalance_price_15

from src.ingestion.entsoe_prices_hourly import main as day_ahead_prices

from src.ingestion.entsoe_solar_15 import main as solar_generation_15
from src.ingestion.entsoe_solar_forecast_15 import main as solar_forecast_15

from src.ingestion.entsoe_wind_15 import main as wind_generation_15
from src.ingestion.entsoe_wind_forecast_15 import main as wind_forecast_15

from src.ingestion.entsoe_crossborder_flows_15 import main as crossborder_flows
from src.ingestion.entsoe_installed_capacity_15 import main as installed_capacity

from src.ingestion.tennet_settlement_prices import main as tennet_settlement_prices
from src.ingestion.tennet_frr_activations import main as tennet_frr_activations
from src.ingestion.tennet_settled_imbalance_volumes import main as tennet_siv
from src.ingestion.tennet_merit_order_list import main as tennet_mol
from src.ingestion.tennet_balance_delta_high_res import main as tennet_balance_delta


def run_job(name: str, job) -> None:
    print(f"\n→ {name}")
    job()
    print(f"✓ {name} done")


def main() -> None:
    print("\nRAW DATA INGESTION START\n")

    #run_job("CO2 price", co2_price)
    #run_job("Gas price", gas_price)

    #run_job("Weather forecast", weather_forecast)
    #run_job("Weather actual", weather_actual)

    #run_job("ENTSO-E load", load_15)
    #run_job("ENTSO-E load forecast", load_forecast_15)

    #run_job("ENTSO-E imbalance price", imbalance_price_15)

    run_job("TenneT settlement prices", tennet_settlement_prices)
    run_job("TenneT FRR activations", tennet_frr_activations)
    run_job("TenneT settled imbalance volumes", tennet_siv)
    run_job("TenneT merit order list", tennet_mol)
    run_job("TenneT balance delta high-res", tennet_balance_delta)

    #run_job("ENTSO-E day-ahead prices", day_ahead_prices)

    #run_job("ENTSO-E solar generation", solar_generation_15)
    #run_job("ENTSO-E solar forecast", solar_forecast_15)

    #run_job("ENTSO-E wind generation", wind_generation_15)
    #run_job("ENTSO-E wind forecast", wind_forecast_15)

    #run_job("ENTSO-E cross-border flows", crossborder_flows)
    #run_job("ENTSO-E installed capacity", installed_capacity)

    print("\nRAW DATA INGESTION DONE\n")


if __name__ == "__main__":
    main()
