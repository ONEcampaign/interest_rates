import pandas as pd
from bblocks import add_iso_codes_column, set_bblocks_data_path

from scripts.config import Paths
from scripts.debt.clean_data import get_clean_data
from scripts.debt.tools import (
    add_weights,
    calculate_interest_payments,
    compute_grouping_stats,
    compute_weighted_averages,
    keep_market_access_only,
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
    }


def get_average_interest(
    start_year: int,
    end_year: int,
    filter_counterparts: bool = True,
    update_data: bool = False,
) -> pd.DataFrame:
    """Get data with the weighted average interest rate for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=INTEREST_RATE_INDICATOR,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
        update_data=update_data,
    )


def get_interest_payments(
    start_year: int,
    end_year: int,
    filter_counterparts: bool = True,
    update_data: bool = False,
) -> pd.DataFrame:
    """Get data with the interest payments for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=INTEREST_PAYMENTS_INDICATORS,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
        update_data=update_data,
    )


def get_commitments(
    start_year: int,
    end_year: int,
    filter_counterparts: bool = True,
    update_data: bool = False,
) -> pd.DataFrame:
    """Get data with the commitments for each country/counterpart_area pair."""
    data = get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=COMMITMENTS_INDICATORS,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
        update_data=update_data,
    )

    return data


def get_maturities(
    start_year: int,
    end_year: int,
    filter_counterparts: bool = True,
    update_data: bool = False,
) -> pd.DataFrame:
    """Get data with the maturities for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=MATURITY_INDICATOR,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
        update_data=update_data,
    )


def get_grace(
    start_year: int,
    end_year: int,
    filter_counterparts: bool = True,
    update_data: bool = False,
) -> pd.DataFrame:
    """Get data with the grace period for each country/counterpart_area pair."""
    return get_clean_data(
        start_year=start_year,
        end_year=end_year,
        indicators=GRACE_PERIOD_INDICATOR,
        filter_counterparts=filter_counterparts,
        counterparts=study_counterparts(),
        update_data=update_data,
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
    start_year: int,
    end_year: int,
    filter_counterparts: bool = True,
    update_data: bool = False,
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

    rate = get_average_interest(start_year, end_year, filter_counterparts, update_data)
    commitments = get_commitments(
        start_year, end_year, filter_counterparts, update_data
    )
    grace = get_grace(start_year, end_year, filter_counterparts, update_data)
    maturities = get_maturities(start_year, end_year, filter_counterparts, update_data)

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
    discount_rate: float = 0.0,
    new_interest_rate: float | None = None,
    interest_rate_difference: float | None = None,
    *,
    filter_counterparts: bool = True,
    filter_countries: bool = False,
    filter_type: str = None,
    filter_values: str | list[str] = None,
    market_access_only: bool = False,
    add_aggregate: bool = False,
    aggregate_name: str = None,
    only_aggregate: bool = False,
    weights_by: list[str] | None = None,
    update_data: bool = False,
) -> pd.DataFrame:
    """Compute the expected interest payments on new debt for each country/counterpart_area pair.

    A starting year and an end year are required. The data is filtered by default on the
    counterparts that are studied in the paper.

    The discount rate is the discount rate used to compute the present value of the expected
    interest payments. The discount rate is expressed as a percentage.

    The new interest rate is an optional interest rate that is used to compute the expected
    interest payments. If it is not provided, then the actual interest rate is used.
    The new interest rate is expressed as a percentage.

    The interest rate difference is an optional interest rate difference that is used to compute
    the expected interest payments. It is added or subtracted from the actual interest rate.
    If it is not provided, then the actual interest rate is used.
    The interest rate difference is expressed as a percentage.

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

    if weights_by is None:
        weights_idx = [
            "year",
            "income_level",
            "continent",
            "country",
            "counterpart_area",
        ]
    else:
        weights_idx = weights_by

    # Create empty df for grouped data in case it is needed
    group_tot = pd.DataFrame()

    # Get the data
    df = get_merged_rates_commitments_grace_maturities_data(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=filter_counterparts,
        update_data=update_data,
    )

    if filter_countries:
        df = df.loc[lambda d: d[filter_type].isin(filter_values)].reset_index(drop=True)
    if market_access_only:
        df = keep_market_access_only(df)

    # Add expected payments
    df = df.assign(
        expected_payments=df.apply(
            calculate_interest_payments,
            discount_rate=discount_rate,
            new_rate=new_interest_rate,
            rate_difference=interest_rate_difference,
            axis=1,
        )
    )

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
    df = add_weights(df, idx=weights_idx, value_column="value_commitments")

    # Compute weighted average
    df = compute_weighted_averages(
        df, idx=["year", "income_level", "continent", "country", "counterpart_area"]
    )

    df = pd.concat([group_tot, df], ignore_index=True)

    return df


def expected_payment_single_counterpart(
    start_year: int,
    end_year: int,
    counterpart: str,
    discount_rate: float,
    *,
    new_interest_rate: float | None = None,
    interest_rate_difference: float | None = None,
    filter_type: str,
    filter_values: str | list,
    aggregate_name: str,
    market_access_only: bool = False,
    update_data: bool = False,
) -> pd.DataFrame:
    """Helper function to get expected payments for a single counterpart.
    It is a thin wrapper around `expected_payments_on_new_debt` that filters
    the data to only include the counterpart of interest and rounds the
    expected payments to billions."""

    return (
        expected_payments_on_new_debt(
            start_year=start_year,
            end_year=end_year,
            discount_rate=discount_rate,
            new_interest_rate=new_interest_rate,
            interest_rate_difference=interest_rate_difference,
            filter_countries=True,
            filter_type=filter_type,
            filter_values=filter_values,
            add_aggregate=True,
            aggregate_name=aggregate_name,
            only_aggregate=True,
            market_access_only=market_access_only,
            update_data=update_data,
        )
        .pipe(add_iso_codes_column, id_column="country", id_type="regex")
        .loc[lambda d: d.counterpart_area.isin([counterpart])]
        .assign(expected_payments=lambda d: round(d.expected_payments / 1e9, 2))
    )


def counterpart_difference(
    start_year: int,
    end_year: int,
    main_counterpart: str,
    comparison_counterpart: str,
    filter_type: str,
    filter_values: str | list,
    aggregate_name: str,
    update_data: bool = False,
):
    """Helper function to get the difference in expected payments for a single
    counterpart at a new interest rate and the current interest rate.

    The output is a long dataframe which has the selected counterpart at the
    original interest rate and at the new interest rate.

    """

    # Get the actual expected payments
    actual = expected_payment_single_counterpart(
        start_year=start_year,
        end_year=end_year,
        counterpart=main_counterpart,
        discount_rate=0.05,
        filter_type=filter_type,
        filter_values=filter_values,
        aggregate_name=aggregate_name,
        market_access_only=False,
        update_data=update_data,
    )

    # Get the rates for the comparison counterpart for the same years
    comparison = expected_payment_single_counterpart(
        start_year=start_year,
        end_year=end_year,
        counterpart=comparison_counterpart,
        discount_rate=0.05,
        filter_type=filter_type,
        filter_values=filter_values,
        aggregate_name=aggregate_name,
        market_access_only=False,
    ).filter(["year", "avg_rate"])

    # Merge the two dataframes
    actual = actual.merge(
        comparison, on="year", how="left", suffixes=("", "_comparison")
    )

    # Calculate the expected payments at the new rate
    with_new_expected = actual.assign(
        expected_payments_at_new_rate=lambda d: d.apply(
            lambda r: expected_payment_single_counterpart(
                start_year=start_year,
                end_year=end_year,
                counterpart=main_counterpart,
                new_interest_rate=r.avg_rate_comparison,
                discount_rate=0.05,
                filter_type=filter_type,
                filter_values=filter_values,
                aggregate_name=aggregate_name,
                market_access_only=False,
            )
            .loc[lambda result: result.year == r.year, "expected_payments"]
            .values[0],
            axis=1,
        )
    )

    return with_new_expected
