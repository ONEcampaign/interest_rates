import pandas as pd

from scripts.config import Paths
from bblocks import set_bblocks_data_path, WorldEconomicOutlook

set_bblocks_data_path(Paths.raw_data)


def get_government_revenue_gdp(update_data: bool = False) -> pd.DataFrame:
    indicator = "GGR_NGDP"
    weo = WorldEconomicOutlook()
    weo.load_data(indicator=indicator)
    if update_data:
        weo.update_data(reload_data=True, year=None, release=None)
    return weo.get_data().assign(
        indicator="Government Revenue (% GDP)", year=lambda d: d.year.dt.year
    )


def get_government_expenditure_gdp(update_data: bool = False) -> pd.DataFrame:
    indicator = "GGX_NGDP"
    weo = WorldEconomicOutlook()
    weo.load_data(indicator=indicator)
    if update_data:
        weo.update_data(reload_data=True, year=None, release=None)
    return weo.get_data().assign(
        indicator="Government Expenditure (% GDP)", year=lambda d: d.year.dt.year
    )


def get_gdp_usd(update_data: bool = False) -> pd.DataFrame:
    indicator = "NGDPD"
    weo = WorldEconomicOutlook()
    weo.load_data(indicator=indicator)

    if update_data:
        weo.update_data(reload_data=True, year=None, release=None)
    return weo.get_data().assign(
        indicator="GDP (USD)",
        value=lambda d: d.value * 1e9,
        year=lambda d: d.year.dt.year,
    )
