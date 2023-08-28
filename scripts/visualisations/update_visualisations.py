import datetime
import json
import os

from bblocks import WFPData, WorldEconomicOutlook

from scripts import config
from scripts.fed_rates.rates_chart import (
    update_fed_rate_hikes_chart_data,
    wide_fed_rates_chart,
)
from scripts.inflation.inflation_charts import inflation_key_numbers
from scripts.logger import logger
from scripts.social_spending.debt_social_chart import debt_health_comparison_chart
from scripts.visualisations.interest_flourish import (
    chart_africa_other_bondholders_ibrd_line,
    chart_data_africa_other_rates_scatter,
    chart_scrolly_bars_africa_bonds_vs_ibrd_rates,
    chart_scrolly_bars_mics_bonds_vs_ibrd_rates,
    chart_scrolly_chart_map_africa_bonds_2021_rates,
    chart_scrolly_chart_map_africa_ibrd_2021_rates,
    export_africa_geometries,
)


def update_key_number(path: str, new_dict: dict) -> None:
    """Update a key number json by updating it with a new dictionary"""

    # Check if the file exists, if not create
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)

    with open(path, "r") as f:
        data = json.load(f)

    for k in new_dict.keys():
        data[k] = new_dict[k]

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------- FED RATES CHART ---------------------- #


def update_fed_charts() -> None:
    update_fed_rate_hikes_chart_data()
    wide_fed_rates_chart()


# ---------------------- INFLATION ---------------------- #
def update_inflation_data() -> None:
    # Update the raw data
    wfp = WFPData()
    wfp.load_data("inflation")
    wfp.update_data(True)

    # Update key numbers
    data = inflation_key_numbers()
    update_key_number(config.Paths.output / "inflation_key_numbers.json", data)


# ---------------------- INTEREST RATES CHART ---------------------- #


def update_interest_data_and_charts() -> None:
    export_africa_geometries()
    chart_scrolly_bars_africa_bonds_vs_ibrd_rates(update_data=True)
    chart_scrolly_bars_mics_bonds_vs_ibrd_rates()
    chart_africa_other_bondholders_ibrd_line(start_year=2000, end_year=2021)
    chart_data_africa_other_rates_scatter(start_year=2000, end_year=2021)
    chart_scrolly_chart_map_africa_ibrd_2021_rates()
    chart_scrolly_chart_map_africa_bonds_2021_rates()


# ---------------------- HEALTH DEBT DATA  ---------------------- #


def update_debt_health_chart_data() -> None:
    indicator = "NGDPD"
    weo = WorldEconomicOutlook()
    weo.load_data(indicator=indicator)
    weo.update_data(reload_data=True, year=None, release=None)
    debt_health_comparison_chart()


def update_visualisations() -> None:
    """Pipeline to update all visualisations"""

    update_fed_charts()
    logger.info("Updated FED charts")

    update_inflation_data()
    logger.info("Updated inflation data")


def update_other_visualisations() -> None:
    """Pipeline to update visualisations with data that is infrequently updated"""
    update_interest_data_and_charts()
    logger.info("Updated interest data and charts")

    update_debt_health_chart_data()
    logger.info("Updated debt health chart data")


if __name__ == "__main__":
    update_visualisations()
    if datetime.datetime.weekday(datetime.datetime.now()) == 0:
        update_other_visualisations()
