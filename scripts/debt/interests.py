import pandas as pd
from bblocks import (
    DebtIDS,
    add_income_level_column,
    convert_id,
    format_number,
    set_bblocks_data_path,
)

from scripts.config import Paths

set_bblocks_data_path(Paths.raw_data)


def interest_indicators() -> list:
    return ["DT.INR.DPPG"]


def maturities_indicators() -> list:
    return [
        "DT.MAT.DPPG",
        # "DT.MAT.OFFT",
        # "DT.MAT.PRVT",
    ]


def study_counterparts() -> list:
    return [
        "Bondholders",
        "World Bank-IDA",
        "World Bank-IBRD",
        "African Dev. Bank",
        "World",
    ]


def _clean_counterpart_area(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        counterpart_area=lambda d: d.counterpart_area.apply(
            lambda r: r.replace("\xa0", "")
        )
    )


def _year2int(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(year=lambda d: d.year.dt.year)


def _clean_indicators(
    df: pd.DataFrame, filter_counterparts: bool = True, filter_columns: bool = True
) -> pd.DataFrame:
    df = (
        df.pipe(_clean_counterpart_area)
        .pipe(add_income_level_column, id_column="country", id_type="regex")
        .pipe(_year2int)
        .sort_values(["country", "year", "counterpart_area"])
        .dropna(subset=["income_level"])
    )

    if filter_counterparts:
        df = df.query(f"counterpart_area in {study_counterparts()}")

    if filter_columns:
        df = df.filter(
            ["country", "counterpart_area", "income_level", "year", "value"], axis=1
        )

    return df.reset_index(drop=True)


def get_average_interest(start_year: int, end_year: int) -> pd.DataFrame:
    indicator = "DT.INR.DPPG"

    ids = (
        DebtIDS()
        .load_data(indicators=indicator, start_year=start_year, end_year=end_year)
        .get_data()
        .pipe(_clean_indicators)
    )

    return ids


def get_interest_payments(start_year: int, end_year: int) -> pd.DataFrame:
    indicators = {
        "DT.INT.BLAT.CD": "Bilateral Interests",
        "DT.INT.MLAT.CD": "Multilateral Interests",
        "DT.INT.PBND.CD": "Private Bonds Interests",
        "DT.INT.PCBK.CD": "Private Banks Interests",
        "DT.INT.PROP.CD": "Private Other Interests",
    }

    ids = (
        DebtIDS()
        .load_data(
            indicators=list(indicators), start_year=start_year, end_year=end_year
        )
        .get_data()
        .pipe(_clean_indicators)
    )

    return ids


def get_commitments(start_year: int, end_year: int) -> pd.DataFrame:
    indicators = {
        "DT.COM.BLAT.CD": "Bilateral commitments",
        "DT.COM.MLAT.CD": "Multilateral commitments",
        "DT.COM.PRVT.CD": "Private commitments",
    }

    ids = (
        DebtIDS()
        .load_data(
            indicators=list(indicators), start_year=start_year, end_year=end_year
        )
        .get_data()
        .pipe(_clean_indicators)
    )

    return ids


def get_maturities(start_year: int, end_year: int) -> pd.DataFrame:
    indicator = maturities_indicators()

    ids = (
        DebtIDS()
        .load_data(indicators=indicator, start_year=start_year, end_year=end_year)
        .get_data()
        .pipe(_clean_indicators)
    )

    return ids


def _order_income(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            order=lambda d: d.income_level.map(
                {
                    "Low income": 1,
                    "Lower middle income": 2,
                    "Upper middle income": 3,
                    "High income": 4,
                }
            )
        )
        .sort_values(
            ["order", "continent", "country", "year", "counterpart_area"],
            ascending=(True, False, True, True, True),
        )
        .drop(columns=["order"])
        .reset_index(drop=True)
    )


def _add_continent(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        continent=lambda d: convert_id(
            d.country, from_type="regex", to_type="continent"
        )
    )


def _flag_africa(df: pd.DataFrame, target_column: str = "continent") -> pd.DataFrame:
    return df.assign(
        **{
            target_column: lambda d: d.continent.map({"Africa": "Africa"}).fillna(
                "Other"
            )
        }
    )


def add_weights(df: pd.DataFrame, idx: list = None) -> pd.DataFrame:
    if idx is None:
        idx = ["year", "counterpart_area"]
    return df.assign(
        weight=lambda d: d.groupby(idx, group_keys=False).value.transform(
            lambda r: r / r.sum()
        )
    )


def avg_africa_interest_rate(
    start_year: int = 2000, end_year: int = 2021
) -> pd.DataFrame:
    rate = get_average_interest(start_year, end_year)
    commitments = get_commitments(start_year, end_year)

    idx = ["country", "counterpart_area", "year", "income_level"]

    df = pd.merge(rate, commitments, on=idx, how="left", suffixes=("", "_commitments"))

    # Keep only rows with positive commitments
    df = df.loc[lambda d: d.value_commitments > 0]

    # Add continent and keep only Africa
    df = df.pipe(_add_continent).loc[lambda d: d.continent == "Africa"]

    # Add weights
    df = add_weights(df)

    # Compute weighted average
    df = (
        df.assign(avg_interest=lambda d: d.value * d.weight)
        .groupby(
            ["year", "counterpart_area", "continent"], as_index=False, dropna=False
        )["avg_interest"]
        .sum()
    )

    return df.pivot(
        index=["year", "continent"], columns="counterpart_area", values="avg_interest"
    ).reset_index()


def scatter_rate_interest(start_year: int = 2000, end_year: int = 2021) -> pd.DataFrame:
    # Get data
    rate = get_average_interest(start_year, end_year)
    interest = get_interest_payments(start_year, end_year)
    commitments = get_commitments(start_year, end_year)

    idx = ["country", "counterpart_area", "year", "income_level"]

    # merge data
    df = pd.merge(rate, interest, on=idx, how="left", suffixes=("", "_interest")).merge(
        commitments, on=idx, how="left", suffixes=("", "_commitments")
    )

    # Keep only rows with positive commitments
    df = df.loc[lambda d: d.value_commitments > 0]

    # Add continent
    df = df.pipe(_add_continent)

    # Flag Africa
    df = df.pipe(_flag_africa)

    # Order income
    df = df.pipe(_order_income)

    # format tooltip numbers
    df = df.assign(
        value_interest_l=lambda d: format_number(
            d.value_interest, as_millions=True, decimals=1
        ),
        value_commitments=lambda d: format_number(
            d.value_commitments, as_millions=True, decimals=2
        ),
    ).drop(columns=["value_interest"])

    return df


if __name__ == "__main__":
    # scatter = scatter_rate_interest()
    # scatter.to_clipboard(index=False)
    ...
    d = get_maturities(2020, 2021)
    d.to_clipboard(index=False)
