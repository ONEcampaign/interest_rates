import pandas as pd
from bblocks import add_iso_codes_column, format_number

from scripts.config import Paths
from scripts.debt.interest_analysis import (
    expected_payments_on_new_debt,
    get_merged_rates_commitments_payments_data,
)
from scripts.debt.tools import (
    add_weights,
    compute_weighted_averages,
    flag_africa,
    keep_market_access_only,
    order_income,
)


def scatter_rate_interest_africa_other(
    start_year: int = 2000,
    end_year: int = 2021,
    filter_counterparts: bool = True,
) -> pd.DataFrame:
    """Data for a scatterplot of interest rates for africa and other countries"""
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
    ]

    return df.filter(output_cols, axis=1)


def chart_data_africa_other_rates_scatter(start_year: int, end_year: int) -> None:
    """A CSV of the data for a scatterplot of interest rates for africa and other countries.

    NOTE: This chart is not used on the final, live analysis.
    """
    afr_others_rates_scatter = scatter_rate_interest_africa_other(
        start_year=start_year, end_year=end_year
    )
    afr_others_rates_scatter.to_csv(
        Paths.output / f"afr_others_rates_scatter_{start_year}_{end_year}.csv",
        index=False,
    )


def smooth_line_rate_interest_africa_other_income(
    start_year: int, end_year: int, filter_counterparts: bool = True
) -> pd.DataFrame:
    """Data for a smooth line of interest rates for africa and other countries"""

    # Get data
    df = get_merged_rates_commitments_payments_data(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=filter_counterparts,
    )

    # Flag Africa
    df = df.pipe(flag_africa)

    # Keep only countries with market access
    # df = df.pipe(keep_market_access_only)

    # Add weights
    idx = ["year", "counterpart_area", "continent", "income_level"]
    df = add_weights(df, idx=idx, value_column="value_commitments")

    # Compute weighted average
    df = df.pipe(compute_weighted_averages, idx=idx, value_columns=["value_rate"])

    # Filter columns
    cols = ["year", "counterpart_area", "income_level", "continent", "avg_rate"]
    df = df.filter(cols, axis=1)

    # pivot continent
    df = df.pivot(
        index=["year", "counterpart_area", "income_level"],
        columns="continent",
        values="avg_rate",
    ).reset_index()

    # Order income
    df = df.pipe(
        order_income,
        idx=["order", "counterpart_area", "income_level", "year"],
        order=[True, False, True, False],
    )

    df["Africa"] = df["Africa"].round(3)
    df["Other"] = df["Other"].round(3)

    return df


def chart_africa_other_bondholders_ibrd_line(start_year: int, end_year: int) -> None:
    """A CSV of the data for a smooth line of interest rates for africa and other countries.

    NOTE: This chart IS USED in the final, live analysis.

    """

    afr_others_rates_smooth_line = (
        smooth_line_rate_interest_africa_other_income(
            start_year=start_year, end_year=end_year
        )
        .loc[lambda d: d.counterpart_area.isin(["World Bank-IBRD", "Bondholders"])]
        .melt(
            id_vars=["year", "counterpart_area", "income_level"],
            var_name="debtor",
            value_name="avg_rate",
        )
        .pivot(
            index=["year", "debtor", "income_level"],
            columns="counterpart_area",
            values="avg_rate",
        )
        .reset_index()
        .pipe(
            order_income,
            ["order", "debtor", "income_level", "year"],
            order=[True, True, True, True],
        )
    )

    afr_others_rates_smooth_line.to_csv(
        Paths.output / f"afr_others_rates_smooth_line_{start_year}_{end_year}.csv",
        index=False,
    )


def export_africa_geometries():
    """Export a CSV of the geometries for African countries.

    NOTE: This is used by the scrolly map showing IBRD and Bond rates for Africa.

    """
    from country_converter import CountryConverter
    from bblocks.dataframe_tools.add import add_flourish_geometries

    africa = (
        CountryConverter()
        .data[["ISO3", "continent"]]
        .loc[lambda d: d.continent == "Africa"]
    )

    africa = add_flourish_geometries(africa, "ISO3", "ISO3")

    africa.filter(["ISO3", "geometry"]).to_csv(
        Paths.output / "africa_geometries.csv", index=False
    )


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
    market_access_only: bool = True,
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
        )
        .pipe(add_iso_codes_column, id_column="country", id_type="regex")
        .loc[lambda d: d.counterpart_area.isin([counterpart])]
        .assign(expected_payments=lambda d: round(d.expected_payments / 1e9, 2))
    )


def counterpart_difference(
    counterpart: str,
    new_rate: float,
    filter_type: str,
    filter_values: str | list,
    aggregate_name: str,
):
    """Helper function to get the difference in expected payments for a single
    counterpart at a new interest rate and the current interest rate.

    The output is a long dataframe which has the selected counterpart at the
    original interest rate and at the new interest rate.

    """

    actual = expected_payment_single_counterpart(
        start_year=2016,
        end_year=2021,
        counterpart=counterpart,
        discount_rate=0.05,
        filter_type=filter_type,
        filter_values=filter_values,
        aggregate_name=aggregate_name,
        market_access_only=False,
    )

    at_new_rate = expected_payment_single_counterpart(
        start_year=2016,
        end_year=2021,
        counterpart=counterpart,
        discount_rate=0.05,
        new_interest_rate=new_rate,
        filter_type=filter_type,
        filter_values=filter_values,
        aggregate_name=aggregate_name,
        market_access_only=False,
    ).assign(counterpart_area=f"{counterpart} at new rate", avg_rate=new_rate)

    return pd.concat([actual, at_new_rate], ignore_index=True)


if __name__ == "__main__":
    ...
