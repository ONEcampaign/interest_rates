import numpy as np
from bblocks import (
    set_bblocks_data_path,
    WFPData,
    add_short_names_column,
    convert_id,
    filter_african_countries,
    WorldEconomicOutlook,
)
from scripts.config import Paths
import pandas as pd

set_bblocks_data_path(Paths.raw_data / "bblocks_data")


def _weo_advanced_economies() -> list:
    return convert_id(
        pd.Series(
            [
                "Andorra",
                "Australia",
                "Austria",
                "Belgium",
                "Canada",
                "Croatia",
                "Cyprus",
                "Czech Republic",
                "Denmark",
                "Estonia",
                "Finland",
                "France",
                "Germany",
                "Greece",
                "Hong Kong SAR",
                "Iceland",
                "Ireland",
                "Israel",
                "Italy",
                "Japan",
                "Korea",
                "Latvia",
                "Lithuania",
                "Luxembourg",
                "Macao SAR",
                "Malta",
                "The Netherlands",
                "New Zealand",
                "Norway",
                "Portugal",
                "Puerto Rico",
                "San Marino",
                "Singapore",
                "Slovak Republic",
                "Slovenia",
                "Spain",
                "Sweden",
                "Switzerland",
                "Taiwan Province of China",
                "United Kingdom",
                "United States",
            ],
            name="name_short",
        )
    ).to_list()


def _world_inflation(wfp: WFPData, indicator="Inflation Rate") -> pd.DataFrame:
    return (
        wfp.get_data("inflation")
        .pipe(add_short_names_column, id_column="iso_code")
        .loc[lambda d: d.date.dt.year.between(2019, 2023)]
        .loc[lambda d: d.indicator == indicator]
        .loc[lambda d: d.iso_code != "VEN"]
        .filter(["name_short", "iso_code", "date", "indicator", "value"], axis=1)
        .rename(
            columns={
                "indicator": "indicator_name",
            }
        )
    )


def _ppp_gdp() -> pd.DataFrame:
    weo = WorldEconomicOutlook()
    weo.load_data("PPPGDP")
    return (
        weo.get_data().assign(year=lambda d: d.year.dt.year).drop(columns=["indicator"])
    )


def __calc_weighted_average(group, threshold: int = 100):
    group = group.dropna(subset=["value", "value_ppp"], how="any")
    if group.iso_code.count() < threshold:
        return pd.Series([np.nan], index=["value"])

    group = group.assign(weight=lambda d: d.value_ppp / d.value_ppp.sum())

    result = round((group["value"] * group["weight"]).sum(), 1)

    return pd.Series([result], index=["value"])


def _calculate_world_weighted_average(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["date", "indicator_name"], as_index=False, dropna=False, observed=True
        )
        .apply(__calc_weighted_average, 145)
        .assign(name_short="World")
    )


def _calculate_africa_median(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(filter_african_countries, id_column="name_short")
        .groupby(
            ["date", "indicator_name"],
            as_index=False,
            dropna=False,
            observed=True,
        )
        .apply(__calc_weighted_average, 48)
        .assign(name_short="Africa")
    )


def _get_latest(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["value"]).loc[
        lambda d: d.date == d.date.max(), ["value", "date"]
    ]


def _get_max(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["value"]).loc[
        lambda d: d.value == d.value.max(), ["value", "date"]
    ]


def inflation_key_numbers():
    wfp = WFPData()
    wfp.load_data("inflation")

    data = _world_inflation(wfp)
    ppg_gdp = _ppp_gdp()

    data = (
        data.assign(year=lambda d: d.date.dt.year)
        .merge(
            ppg_gdp.filter(["year", "iso_code", "value"]),
            how="left",
            on=["year", "iso_code"],
            suffixes=("", "_ppp"),
        )
        .drop(columns=["year"])
    )

    world = _calculate_world_weighted_average(data).dropna(subset=["value"])
    africa = _calculate_africa_median(data).dropna(subset=["value"])

    world_max = _get_max(world)
    africa_max = _get_max(africa)

    world_latest = _get_latest(world)
    africa_latest = _get_latest(africa)

    return {
        "world_max_value": world_max["value"].values[0],
        "world_max_date": world_max["date"].max().strftime("%B %Y"),
        "africa_max_value": africa_max["value"].values[0],
        "africa_max_date": africa_max["date"].max().strftime("%B %Y"),
        "world_latest_value": world_latest["value"].values[0],
        "world_latest_date": world_latest["date"].max().strftime("%B %Y"),
        "africa_latest_date": africa_latest["date"].max().strftime("%B %Y"),
        "africa_latest_value": africa_latest["value"].max(),
    }


if __name__ == "__main__":
    data = inflation_key_numbers()
