import pandas as pd
from bblocks import (
    add_iso_codes_column,
    set_bblocks_data_path,
    DebtIDS,
    WorldEconomicOutlook,
)
from bblocks.dataframe_tools.add import add_gdp_column

from scripts.config import Paths

set_bblocks_data_path(Paths.raw_data)

indicators = {
    "commitments_ppg": "DT.COM.DPPG.CD",
    "principal_repayments": "DT.AMT.DPPG.CD",
    "principal_forgiven": "DT.AXF.DPPG.CD",
    "debt_stocks": "DT.DOD.DPPG.CD",
    "stock_change": "DT.DOD.DECT.CD.CG",
}

ids = DebtIDS().load_data(
    indicators=list(indicators.values()), start_year=2010, end_year=2021
)

df = (
    ids.get_data()
    .assign(series=lambda d: d.series_code.map({v: k for k, v in indicators.items()}))
    .drop("series_code", axis=1)
    .loc[lambda d: d.counterpart_area == "World"]
    .pivot(
        index=["country", "counterpart_area", "year"], columns="series", values="value"
    )
    .fillna(0)
    .reset_index()
    .assign(
        net_commitments_repayments=lambda d: d.commitments_ppg - d.principal_repayments
    )
    .sort_values(by=["country", "year"])
    .assign(
        yoy_stock_diff=lambda d: d.groupby("country")["debt_stocks"].diff().shift(-1)
    )
    .loc[lambda d: d.year.dt.year > 2010]
    .assign(unexplained_diff=lambda d: d.yoy_stock_diff - d.net_commitments_repayments)
    .filter(
        [
            "country",
            "year",
            "debt_stocks",
            "commitments_ppg",
            "principal_repayments",
            "net_commitments_repayments",
            "yoy_stock_diff",
            "unexplained_diff",
            "principal_forgiven",
            "stock_change",
        ]
    )
    .pipe(
        add_gdp_column,
        id_column="country",
        id_type="regex",
        date_column="year",
        include_estimates=True,
    )
    .pipe(add_iso_codes_column, id_column="country", id_type="regex")
    .loc[lambda d: d.country == "El Salvador"]
)

WEO_INDICATORS = {
    "NGDP_D": "gdp_deflator",
    "NGDP": "gdp_domestic",
    "NGDPD": "gdp_usd",
}

weo = WorldEconomicOutlook().load_data(list(WEO_INDICATORS))

weo_df = (
    weo.get_data()
    .pivot(index=["iso_code", "year"], columns="indicator", values="value")
    .reset_index()
    .assign(
        index_year=lambda d: d.groupby("iso_code")["year"].transform(
            lambda x: x.loc[d["NGDP_D"].round(0) == 100].max()
        )
    )
    .assign(
        gdp_def2020=lambda d: d["NGDP_D"]
        / d.groupby("iso_code")["NGDP_D"].transform(
            lambda x: x.loc[d["year"].dt.year == 2020].max()
        )
    )
    .loc[lambda d: d.year.dt.year.between(2010, 2022)]
)


df = df.merge(weo_df, on=["iso_code", "year"], how="left")
