import pandas as pd

from scripts.config import Paths
from scripts.debt.interest_analysis import expected_payments_on_new_debt


def base_data_loans_observable_by_country_group_year(start_year: int, end_year: int):
    """Data for Observable. It is broken down by country for the basic loan
    information, and includes group names (and weights by group/counterpart). This data
    is used in the notebook to produce the interactive charts.
    """

    # Data for the "Africa" group.
    africa_data = expected_payments_on_new_debt(
        start_year=start_year,
        end_year=end_year,
        filter_countries=True,
        filter_type="continent",
        filter_values="Africa",
        weights_by=["year", "counterpart_area", "continent"],
    ).assign(group_name="Africa")

    # Data for the "Middle income countries" group"
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


def chart_observable_interactive_interest_payments() -> None:
    """A CSV of the data for the interactive chart of interest payments."""
    df = base_data_loans_observable_by_country_group_year(
        start_year=2017, end_year=2021
    )
    df.to_csv(
        Paths().output / "country_counterpart_with_weights_2017-21.csv.csv", index=False
    )


if __name__ == "__main__":
    chart_observable_interactive_interest_payments()
