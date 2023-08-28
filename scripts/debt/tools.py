import numpy as np
import pandas as pd


import logging

logging.getLogger("country_converter").setLevel(logging.ERROR)


def order_income(
    df: pd.DataFrame, idx: list = None, order: list = None
) -> pd.DataFrame:
    """Order a DataFrame by income level.

    Optionally specify the columns to use for ordering
    and the order (ascending or descending).
    """
    income_order = {
        "Low income": 1,
        "Lower middle income": 2,
        "Upper middle income": 3,
        "High income": 4,
    }
    if idx is None:
        idx = ["order", "counterpart_area", "continent", "country", "year"]

    if order is None:
        order = [True, False, True, True, False]

    return (
        df.assign(order=lambda d: d.income_level.map(income_order))
        .sort_values(idx, ascending=order)
        .drop(columns=["order"])
        .reset_index(drop=True)
    )


def flag_africa(df: pd.DataFrame, target_column: str = "continent") -> pd.DataFrame:
    """Flag Africa (as a continent) in a DataFrame. This is done by assigning Africa
    or Other to the target column"""
    return df.assign(
        **{
            target_column: lambda d: d.continent.map({"Africa": "Africa"}).fillna(
                "Other"
            )
        }
    )


def add_weights(
    df: pd.DataFrame, idx: list = None, value_column: str = "value"
) -> pd.DataFrame:
    """Add weights to a DataFrame. The weights are calculated as the share of the
    value_column in the group defined by idx.
    """

    # If no index is specified, use the following columns
    if idx is None:
        idx = ["year", "counterpart_area"]

    # Calculate the weights
    df = df.assign(
        weight=lambda d: d.groupby(idx, group_keys=False)[value_column].transform(
            lambda r: r / r.sum()
        )
    )

    return df


def calculate_interest_payments(
    row: pd.Series,
    discount_rate: float = 0.0,
    new_rate: float = None,
    rate_difference: float = None,
) -> float:
    """Calculate the NPV of the interest payments for a given row of a DataFrame. The row must
    contain the following columns:

    - value_commitments.
    - value_maturities.
    - value_grace.
    - value_rate.

    The NPV of interest payments are calculated as follows:
    - NPV of interest payments during grace period.
    - NPV of interest payments after grace period.
    - total NPV of interest payments.

    A discount rate of 0.0 is used by default. This means calculating payments in nominal terms.
    """

    # Calculate the number of years in which principal will be paid
    payment_years = row.value_maturities - row.value_grace

    # Calculate the principal payment per year
    if payment_years <= 0:
        principal_payment_per_year = 0
    else:
        principal_payment_per_year = row.value_commitments / payment_years

    if new_rate is not None:
        row.value_rate = new_rate

    if rate_difference is not None:
        row.value_rate = row.value_rate + rate_difference

    # Since the rate is given in percentage points, we need to divide by 100
    try:
        rate = row.value_rate / 100
    except ZeroDivisionError:
        rate = 0

    # Calculate interests during grace period. Discount them to present value.
    grace_period_interest = 0
    for year in range(1, int(np.floor(row.value_grace)) + 1):
        # calculate the payment amoung
        payment_amount = row.value_commitments * rate

        # Calculate the discount factor
        discount_factor = (1 + discount_rate) ** year

        # Calculate the (discounted) interest payment for the year
        grace_period_interest += payment_amount / discount_factor

    # Calculate the interest payments for each year after grace
    loan_interests_after_grace = 0
    for year in range(1, int(np.ceil(payment_years))):
        # Calculate the payment amount
        payment_amount = (
            row.value_commitments - year * principal_payment_per_year
        ) * rate

        # Calculate discount factor
        discount_factor = (1 + discount_rate) ** (year + row.value_grace)

        # Calculate the (discounted) interest payment for the year
        loan_interests_after_grace += payment_amount / discount_factor

    # Put everything together
    total_interest = grace_period_interest + loan_interests_after_grace

    return total_interest


def compute_weighted_averages(
    df: pd.DataFrame, idx: list = None, value_columns: list = None
) -> pd.DataFrame:
    """Compute the weighted average of the value_columns for each group defined by idx.

    This is done based on the "weight" column of the dataframe. This must be computed
    before calling this function.

    """

    # Set default value for idx
    if idx is None:
        idx = ["year", "country", "counterpart_area"]

    # Set default value for value_columns
    if value_columns is None:
        value_columns = ["value_rate", "value_maturities", "value_grace"]

    # Compute weighted average and group by index
    for col in value_columns:
        df[f"avg_{col.split('_')[1]}"] = df[col] * df.weight

    # Compute weighted average and group by index
    df = df.groupby(idx, as_index=False, dropna=False, observed=True).sum(
        numeric_only=True
    )

    return df


def compute_grouping_stats(
    df: pd.DataFrame,
    filter_type: str,
    filter_values: list[str],
    group_name: str,
    idx: list = None,
    value_columns: list = None,
) -> pd.DataFrame:
    """Compute the weighted averages for a group of countries.

    To do that, the data is filtered to keep a specific group of countries,
    and the data is weighted and then aggregated (sum) to have a single value
    for the group according to the index provided.

    The group is defined by the groupby_type and the group name.
    The groupby_type must be either "continent" or "income_level".
    The group name must be one of the values in the groupby_type column.

    The idx parameter is the list of columns to use to group the data.
    The value_columns parameter is the list of columns to compute the weighted averages on.
    """

    if filter_type not in ["continent", "income_level", "country"]:
        raise ValueError("groupby_type must be either 'continent' or 'income_level'")

    if idx is None:
        idx = ["year", "counterpart_area"]

    # Create a copy of the data
    group_data = df.copy(deep=True)

    # filter the data to keep only the relevant data for the group
    group_data = group_data.loc[lambda d: d[filter_type].isin(filter_values)]

    # Compute the weights based on commitments
    group_data = add_weights(group_data, idx=idx, value_column="value_commitments")

    # Compute the weighted averages
    group_data = compute_weighted_averages(
        group_data, idx=idx, value_columns=value_columns
    )

    # Add the group name
    group_data = group_data.assign(country=group_name)

    return group_data


def keep_market_access_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out countries without market access"""
    # Keep only countries with market access
    market_countries = df.query(
        "counterpart_area == 'Bondholders' and value_commitments.notna()"
    ).country.unique()

    df = df.loc[lambda d: d.country.isin(market_countries)]

    return df.reset_index(drop=True)
