import pandas as pd
from bblocks import add_iso_codes_column, format_number

from scripts.config import Paths
from scripts.debt.interest_analysis import (
    counterpart_difference,
    expected_payments_on_new_debt,
    get_merged_rates_commitments_payments_data,
)
from scripts.debt.tools import (
    add_weights,
    compute_weighted_averages,
    flag_africa,
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


def _helper_scrolly_chart_map_counterpart_2021_rates(
    counterpart: str = "World Bank-IBRD",
) -> pd.DataFrame:
    """A CSV of the data for a scrolly map of IBRD rates"""

    return (
        expected_payments_on_new_debt(
            start_year=2021,
            end_year=2021,
            discount_rate=0.05,
            filter_type="continent",
            filter_values="Africa",
            filter_countries=True,
            market_access_only=False,
            add_aggregate=False,
        )
        .loc[lambda d: d.counterpart_area == counterpart]
        .pipe(add_iso_codes_column, id_column="country", id_type="regex")
        .filter(["iso_code", "country", "year", "value_rate", "continent"])
        .rename(columns={"value_rate": "rate"})
    )


def chart_scrolly_chart_map_africa_ibrd_2021_rates() -> None:
    data = _helper_scrolly_chart_map_counterpart_2021_rates("World Bank-IBRD")

    data.to_csv(
        Paths.output / "scrolly_chart_map_ibrd_africa_2021_rates.csv", index=False
    )


def chart_scrolly_chart_map_africa_bonds_2021_rates() -> None:
    data = _helper_scrolly_chart_map_counterpart_2021_rates("Bondholders")

    data.to_csv(
        Paths.output / "scrolly_chart_map_ibrd_africa_2021_rates.csv", index=False
    )


def chart_scrolly_bars_africa_bonds_vs_ibrd_rates() -> None:
    data = counterpart_difference(
        start_year=2017,
        end_year=2021,
        main_counterpart="Bondholders",
        comparison_counterpart="World Bank-IBRD",
        filter_type="continent",
        filter_values="Africa",
        aggregate_name="Africa",
    )

    data.filter(
        [
            "year",
            "country",
            "counterpart_area",
            "expected_payments",
            "expected_payments_at_new_rate",
        ]
    ).to_csv(
        Paths.output / "scrolly_bars_africa_bonds_vs_at_ibrd_rates.csv", index=False
    )


def chart_scrolly_bars_mics_bonds_vs_ibrd_rates() -> None:
    data = counterpart_difference(
        start_year=2017,
        end_year=2021,
        main_counterpart="Bondholders",
        comparison_counterpart="World Bank-IBRD",
        filter_type="income_level",
        filter_values=["Lower middle income", "Upper middle income"],
        aggregate_name="Middle income countries",
    )

    data.filter(
        [
            "year",
            "country",
            "counterpart_area",
            "expected_payments",
            "expected_payments_at_new_rate",
        ]
    ).to_csv(Paths.output / "scrolly_bars_mics_bonds_vs_at_ibrd_rates.csv", index=False)


if __name__ == "__main__":
    export_africa_geometries()
    chart_africa_other_bondholders_ibrd_line(start_year=2000, end_year=2021)
    chart_data_africa_other_rates_scatter(start_year=2000, end_year=2021)
    chart_scrolly_chart_map_africa_ibrd_2021_rates()
    chart_scrolly_chart_map_africa_bonds_2021_rates()
    chart_scrolly_bars_africa_bonds_vs_ibrd_rates()
    chart_scrolly_bars_mics_bonds_vs_ibrd_rates()
