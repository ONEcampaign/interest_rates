from scripts.config import Paths
from scripts.debt.interest_analysis import expected_payments_on_new_debt


def observable_charts_data(start_year: int, end_year: int) -> None:
    columns = [
        "year",
        "counterpart_area",
        "value_commitments",
        "avg_rate",
        "avg_grace",
        "avg_maturities",
        "expected_payments",
        "country",
    ]
    countries_expected_payments = (
        expected_payments_on_new_debt(
            start_year=start_year,
            end_year=end_year,
            filter_counterparts=True,
            filter_type="continent",
            filter_values="Africa",
            filter_countries=True,
            add_aggregate=True,
            aggregate_name="Africa",
        )
        .filter(columns)
        .rename(columns={"avg_maturities": "avg_maturity"})
    )

    countries_expected_payments[["value_commitments", "expected_payments"]] /= 1e6

    countries_expected_payments.to_csv(
        Paths.output / f"expected_payments_{start_year}_{end_year}.csv",
        index=False,
    )

    # Average rates for Africa
    africa_rates = expected_payments_on_new_debt(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=True,
        filter_type="continent",
        filter_values="Africa",
        filter_countries=True,
        add_aggregate=True,
        aggregate_name="Africa",
        only_aggregate=True,
    ).filter(columns)

    africa_rates.to_csv(
        Paths.output / f"africa_overview_{start_year}_{end_year}.csv",
        index=False,
    )

    # Average rates for Africa
    income_rates = expected_payments_on_new_debt(
        start_year=start_year,
        end_year=end_year,
        filter_counterparts=True,
        filter_type="income_level",
        filter_values=["Lower middle income", "Upper middle income"],
        filter_countries=True,
        add_aggregate=True,
        aggregate_name="Middle income",
        only_aggregate=True,
    ).filter(columns)

    income_rates.to_csv(
        Paths.output / f"mics_overview_{start_year}_{end_year}.csv",
        index=False,
    )
