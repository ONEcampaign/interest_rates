import pandas as pd
from bblocks import format_number, set_bblocks_data_path

from scripts.config import Paths
from scripts.debt.clean_data import get_clean_data
from scripts.debt.tools import (
    add_weights,
    calculate_interest_payments,
    compute_grouping_stats,
    compute_weighted_averages,
    flag_africa,
    order_income,
)

set_bblocks_data_path(Paths.raw_data)

INTEREST_RATE_INDICATOR: str = "DT.INR.DPPG"
MATURITY_INDICATOR: str = "DT.MAT.DPPG"
GRACE_PERIOD_INDICATOR: str = "DT.GPA.DPPG"

INTEREST_PAYMENTS_INDICATORS: dict = {
    "DT.INT.BLAT.CD": "Bilateral",
    "DT.INT.MLAT.CD": "Multilateral",
    "DT.INT.PBND.CD": "Private",
    "DT.INT.PCBK.CD": "Private",
    "DT.INT.PROP.CD": "Private",
}

COMMITMENTS_INDICATORS = {
    "DT.COM.BLAT.CD": "Bilateral",
    "DT.COM.MLAT.CD": "Multilateral",
    "DT.COM.PRVT.CD": "Private",
}


def study_counterparts() -> dict:
    """The list of counterparts to keep for the analysis"""
    return {
        "Bondholders": "Private",
        "World Bank-IDA": "Multilateral",
        "World Bank-IBRD": "Multilateral",
        "African Dev. Bank": "Multilateral",
        "European Investment Bank": "Multilateral",
        "European Development Fund (EDF)": "Multilateral",
        "African Export-Import Bank": "Multilateral",
        "European Union": "Multilateral",
    }


def get_average_interest(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Get data with the weighted average interest rate for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=INTEREST_RATE_INDICATOR,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
    )


def get_interest_payments(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Get data with the interest payments for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=INTEREST_PAYMENTS_INDICATORS,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
    )


def get_commitments(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Get data with the commitments for each country/counterpart_area pair."""
    data = get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=COMMITMENTS_INDICATORS,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
    )

    return data


def get_maturities(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Get data with the maturities for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=MATURITY_INDICATOR,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
    )


def get_grace(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Get data with the grace period for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=GRACE_PERIOD_INDICATOR,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
    )


def get_merged_rates_commitments_payments_data(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Get the data with the interest rate, the commitments, the grace period and the maturities.

    the resulting data identifies:
    - interest rates as "value_rate"
    - commitments as "value_commitments"
    - grace period as "value_grace"
    - maturities as "value_maturities"

    The data is merged on the following columns:
    - country
    - counterpart_area
    - continent
    - income_level
    - year

    """

    rate = get_average_interest(start_year, end_year, filter_counterparts)
    commitments = get_commitments(start_year, end_year, filter_counterparts)
    payments = get_interest_payments(start_year, end_year, filter_counterparts)

    idx = ["year", "country", "counterpart_area", "continent", "income_level"]

    # merge the data and keep only rows with positive commitments
    df = (
        pd.merge(commitments, rate, on=idx, how="left", suffixes=("_commitments", ""))
        .merge(payments, on=idx, how="left", suffixes=("", "_payments"))
        .rename(columns={"value": "value_rate"})
        .loc[lambda d: d.value_commitments > 0]
    )

    return df


def get_merged_rates_commitments_grace_maturities_data(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Get the data with the interest rate, the commitments, the grace period and the maturities.

    the resulting data identifies:
    - interest rates as "value_rate"
    - commitments as "value_commitments"
    - grace period as "value_grace"
    - maturities as "value_maturities"

    The data is merged on the following columns:
    - country
    - counterpart_area
    - continent
    - income_level
    - year

    """

    rate = get_average_interest(start_year, end_year, filter_counterparts)
    commitments = get_commitments(start_year, end_year, filter_counterparts)
    grace = get_grace(start_year, end_year, filter_counterparts)
    maturities = get_maturities(start_year, end_year, filter_counterparts)

    idx = ["year", "country", "counterpart_area", "continent", "income_level"]

    # merge the data and keep only rows with positive commitments
    df = (
        pd.merge(commitments, rate, on=idx, how="left", suffixes=("_commitments", ""))
        .merge(grace, on=idx, how="left", suffixes=("", "_grace"))
        .merge(maturities, on=idx, how="left", suffixes=("", "_maturities"))
        .rename(columns={"value": "value_rate"})
        .loc[lambda d: d.value_commitments > 0]
    )

    return df


def expected_payments_on_new_debt(
    start_year: int = 2000,
    end_year: int = 2021,
    *,
    filter_counterparts: bool = True,
    filter_countries: bool = False,
    filter_type: str = None,
    filter_values: str | list[str] = None,
    add_aggregate: bool = False,
    aggregate_name: str = None,
    only_aggregate: bool = False,
) -> pd.DataFrame:
    """Compute the expected interest payments on new debt for each country/counterpart_area pair.

    A starting year and an end year are required. The data is filtered by default on the
    counterparts
    that are studied in the paper.

    If filter_data is True, then the debtor data is filtered on the filter_type and filter_value.
    The filter_type must be one of the following:
    - "continent"
    - "income_level"

    If add_aggregate is True, then the aggregate is calculated and added to the data.
    The aggregate is calculated on the filter_type and filter_value.
    The aggregate_name is the name of the aggregate in the resulting data.

    If only_aggregate is True, then only the aggregate is returned.
    """
    # validate filter values
    if isinstance(filter_values, str):
        filter_values = [filter_values]

    # validate aggregate name
    if add_aggregate and aggregate_name is None:
        raise ValueError("aggregate_name must be provided if add_aggregate is True")

    # Create empty df for grouped data in case it is needed
    group_tot = pd.DataFrame()

    # Get the data
    df = get_merged_rates_commitments_grace_maturities_data(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=filter_counterparts,
    )

    if filter_countries:
        df = df.loc[lambda d: d[filter_type].isin(filter_values)].reset_index(drop=True)

    # Add expected payments
    df = df.assign(expected_payments=df.apply(calculate_interest_payments, axis=1))

    # if add_aggregate then calculate the aggregate
    if add_aggregate:
        group_tot = compute_grouping_stats(
            df=df,
            filter_type=filter_type,
            filter_values=filter_values,
            group_name=aggregate_name,
        ).drop(columns=["value_rate", "value_grace", "value_maturities"])

    if only_aggregate:
        return group_tot

    # Add weights to individual countries
    df = add_weights(
        df,
        idx=["year", "income_level", "continent", "country", "counterpart_area"],
        value_column="value_commitments",
    )

    # Compute weighted average
    df = compute_weighted_averages(
        df, idx=["year", "income_level", "continent", "country", "counterpart_area"]
    )

    df = pd.concat([group_tot, df], ignore_index=True)

    return df


def scatter_rate_interest_africa_other(
    start_year: int = 2000,
    end_year: int = 2021,
    filter_counterparts: bool = True,
) -> pd.DataFrame:

    # Get data
    df = get_merged_rates_commitments_payments_data(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=filter_counterparts,
    )

    # Flag Africa
    df = df.pipe(flag_africa)

    # Order income
    df = df.pipe(order_income)

    # format tooltip numbers
    df = df.assign(
        interest_payments_l=lambda d: format_number(
            d.value_payments, as_millions=True, decimals=1
        ),
        value_commitments=lambda d: format_number(
            d.value_commitments, as_millions=True, decimals=2
        ),
    )

    output_cols = [
        "country",
        "counterpart_area",
        "income_level",
        "year",
        "value_rate",
        "value_commitments",
        "continent",
        "interest_payments_l",
    ]

    return df.filter(output_cols, axis=1)


def smooth_scatter_rate_interest_africa_other(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    # Get data
    df = get_merged_rates_commitments_payments_data(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=filter_counterparts,
    )

    # Flag Africa
    df = df.pipe(flag_africa)

    idx = ["year", "counterpart_area", "continent", "income_level"]

    # Add weights
    df = add_weights(df, idx=idx, value_column="value_commitments")
    df = df.pipe(compute_weighted_averages, idx=idx, value_columns=["value_rate"])

    # Filter columns
    cols = ["year", "counterpart_area", "income_level", "continent", "avg_value_rate"]
    df = df.filter(cols, axis=1)

    # pivot continent
    df = df.pivot(
        index=["year", "counterpart_area", "income_level"],
        columns="continent",
        values="avg_value_rate",
    ).reset_index()

    # Order income
    df = df.pipe(
        order_income,
        ["order", "counterpart_area", "income_level", "year"],
        order=[True, False, True, False],
    )

    return df


def charts_data(start_year: int, end_year: int) -> None:
    afr_others_rates_scatter = scatter_rate_interest_africa_other(
        start_year=start_year, end_year=end_year
    )
    afr_others_rates_scatter.to_csv(
        Paths.output / f"afr_others_rates_scatter_{start_year}_{end_year}.csv",
        index=False,
    )

    afr_others_rates_smooth_scatter = smooth_scatter_rate_interest_africa_other(
        start_year=start_year, end_year=end_year
    )
    afr_others_rates_smooth_scatter.to_csv(
        Paths.output / f"afr_others_rates_smooth_scatter_{start_year}_{end_year}.csv",
    )


if __name__ == "__main__":
    scatter_rate_interest_africa_other(start_year=2000, end_year=2021)
