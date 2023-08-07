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
    df = df.pipe(keep_market_access_only)

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


def borrowing_stats_africa_other(
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

    # Keep only countries with market access
    df = df.pipe(keep_market_access_only)

    # Add weights
    idx = ["year", "counterpart_area", "continent"]
    df = add_weights(df, idx=idx, value_column="value_commitments")

    # Compute weighted average
    df = df.pipe(compute_weighted_averages, idx=idx, value_columns=["value_rate"])

    # Filter columns
    cols = ["year", "counterpart_area", "continent", "avg_rate"]
    df = df.filter(cols, axis=1).assign(avg_rate=lambda d: d.avg_rate.round(2))

    return df


def africa_other_rates_scatter_flourish(start_year: int, end_year: int) -> None:
    afr_others_rates_scatter = scatter_rate_interest_africa_other(
        start_year=start_year, end_year=end_year
    )
    afr_others_rates_scatter.to_csv(
        Paths.output / f"afr_others_rates_scatter_{start_year}_{end_year}.csv",
        index=False,
    )


def africa_other_bondholders_ibrd_line(start_year: int, end_year: int) -> None:
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


def africa_other_borrowing_stats(start_year: int, end_year: int) -> pd.DataFrame:
    borrowing_stats = borrowing_stats_africa_other(
        start_year=start_year, end_year=end_year
    )

    return borrowing_stats


def export_africa_geometries():
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


def explorer_ibrd(start_year: int, end_year: int, filter_counterparts: bool = True):
    df = get_merged_rates_commitments_payments_data(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=filter_counterparts,
    )

    # Flag Africa
    df = df.pipe(flag_africa)

    # Add weights
    idx = ["year", "counterpart_area", "continent", "income_level"]
    df = add_weights(df, idx=idx, value_column="value_commitments")

    df = df.assign(weighted_amount=lambda d: d.value_commitments * d.weight)

    africa_ibrd = (
        df.loc[lambda d: d.counterpart_area.isin(["World Bank-IBRD", "Bondholders"])]
        .loc[lambda d: d.continent == "Africa"]
        .loc[lambda d: d.year == 2021]
        .pipe(add_iso_codes_column, id_column="country", id_type="regex")
        .filter(
            [
                "iso_code",
                "country",
                "counterpart_area",
                "year",
                "value_rate",
                "weighted_amount",
            ]
        )
    )

    africa_ibrd_wide_rate = (
        africa_ibrd.drop(["weighted_amount"], axis=1)
        .pivot(
            index=["iso_code", "country", "year"],
            columns="counterpart_area",
            values="value_rate",
        )
        .reset_index()
        .assign(continent="Africa")
    )

    africa_ibrd_wide_amount = africa_ibrd.drop(["value_rate"], axis=1).assign(
        continent="Africa"
    )

    return africa_ibrd_wide_rate


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
    bonds_actual = expected_payment_single_counterpart(
        start_year=2016,
        end_year=2021,
        counterpart=counterpart,
        discount_rate=0.05,
        filter_type=filter_type,
        filter_values=filter_values,
        aggregate_name=aggregate_name,
        market_access_only=True,
    )

    bonds_ibrd_rate = expected_payment_single_counterpart(
        start_year=2016,
        end_year=2021,
        counterpart=counterpart,
        discount_rate=0.05,
        new_interest_rate=new_rate,
        filter_type=filter_type,
        filter_values=filter_values,
        aggregate_name=aggregate_name,
        market_access_only=True,
    ).assign(counterpart_area=f"{counterpart} at new rate", avg_rate=new_rate)

    return pd.concat([bonds_actual, bonds_ibrd_rate], ignore_index=True)


def african_dev_bank(start_year: int, end_year: int):
    return expected_payment_single_counterpart(
        start_year=start_year,
        end_year=end_year,
        counterpart="African Dev. Bank",
        discount_rate=0.05,
        filter_type="continent",
        filter_values="Africa",
        aggregate_name="Africa",
        market_access_only=True,
    )


def observable_by_country(start_year: int, end_year: int):
    africa_data = expected_payments_on_new_debt(
        start_year=start_year,
        end_year=end_year,
        filter_countries=True,
        filter_type="continent",
        filter_values="Africa",
        weights_by=["year", "counterpart_area", "continent"],
    ).assign(group_name="Africa")

    mic_data = expected_payments_on_new_debt(
        start_year=start_year,
        end_year=end_year,
        filter_countries=True,
        filter_type="income_level",
        filter_values=["Lower middle income", "Upper middle income"],
        weights_by=["year", "counterpart_area"],
    ).assign(group_name="Middle income countries")

    return pd.concat([africa_data, mic_data], ignore_index=True).filter(
        [
            "year",
            "country",
            "group_name",
            "counterpart_area",
            "value_commitments",
            "value_rate",
            "value_grace",
            "value_maturities",
            "expected_payments",
            "avg_rate",
            "avg_maturities",
            "avg_grace",
            "weight",
        ]
    )


if __name__ == "__main__":
    # observable_charts_data(start_year=2000, end_year=2021)
    # flourish_charts_data(start_year=2000, end_year=2021)

    # data = explorer_ibrd(start_year=2000, end_year=2021)
    #
    # bh = data.query("Bondholders.notna()")
    # wb = data.query("`World Bank-IBRD`.notna()")

    # # data_afr = bondholders_difference(new_rate=1.135075)
    # data = counterpart_difference(
    #     counterpart="Bondholders",
    #     new_rate=1.101698,
    #     filter_type="income_level",
    #     filter_values=["Lower middle income", "Upper middle income"],
    #     aggregate_name="MIC",
    # )
    #
    # afr_data = counterpart_difference(
    #     counterpart="Bondholders",
    #     new_rate=1.135075,
    #     filter_type="continent",
    #     filter_values="Africa",
    #     aggregate_name="Africa",
    # )

    data_country = observable_by_country(start_year=2017, end_year=2021)

    data_country.to_csv(
        f"{Paths.output}/country_counterpart_with_weights_2016-21.csv", index=False
    )
