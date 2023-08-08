import pandas as pd
from bblocks import add_income_level_column, add_short_names_column

from scripts import config
from scripts.debt.debt_service import service_data
from scripts.government.revenue import get_gdp_usd, get_government_expenditure_gdp


def debt_gdp(update_data: bool = False) -> pd.DataFrame:
    debt = service_data().assign(year=lambda d: d.year.dt.year)
    gdp = get_gdp_usd(update_data=update_data)
    return (
        pd.merge(debt, gdp, on=["iso_code", "year"], suffixes=("_debt", "_gdp"))
        .assign(
            value=lambda d: round(100 * d.value_debt / d.value_gdp, 4),
            indicator="Debt (% GDP)",
        )
        .filter(["iso_code", "year", "value", "indicator"], axis=1)
    )


def _gdp2exp(indicator_df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
    exp = get_government_expenditure_gdp()
    return (
        pd.merge(
            indicator_df, exp, on=["iso_code", "year"], suffixes=("_indicator", "_exp")
        )
        .assign(
            value=lambda d: round(100 * d.value_indicator / d.value_exp, 4),
            indicator=indicator_name,
        )
        .filter(["iso_code", "year", "value", "indicator"], axis=1)
    )


def debt_exp(update_data: bool = False) -> pd.DataFrame:
    debt = debt_gdp(update_data=update_data)

    return _gdp2exp(debt, "Debt (% Expenditure)")


def health_spending() -> pd.DataFrame:
    health = pd.read_csv(config.Paths.raw_data / "health_spending_gdp.csv")

    return _gdp2exp(health, "Health (% Expenditure)")


def education_spending() -> pd.DataFrame:
    education = pd.read_csv(config.Paths.raw_data / "education_spending_gdp.csv")

    return _gdp2exp(education, "Education (% Expenditure)")


def debt_health_comparison_chart(update_data: bool = False) -> None:
    debt = debt_exp(update_data=update_data)
    health = health_spending()

    df = (
        pd.merge(debt, health, on=["iso_code", "year"], suffixes=("_debt", "_health"))
        .pipe(
            add_short_names_column,
            id_column="iso_code",
            id_type="ISO3",
            target_column="name",
        )
        .filter(
            [
                "iso_code",
                "name",
                "year",
                "value_debt",
                "value_health",
                "value_education",
            ]
        )
    )

    df = add_income_level_column(df, id_column="iso_code", id_type="ISO3")

    # Define labels
    labels = ["very low", "low", "moderate", "high", "very high"]

    order = {
        "very low": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "very high": 4,
    }

    # Compute the quantile thresholds based on the entire dataset's distribution
    quantiles = df["value_debt"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values

    # For each country, compute the median value over the years
    medians = df.groupby("name")["value_debt"].median()

    # Use the computed quantile thresholds to assign a label to each country's median value
    country_labels = pd.cut(medians, bins=quantiles, labels=labels, include_lowest=True)

    # Map the country labels to the original DataFrame
    df["category"] = df["name"].map(country_labels)
    df = (
        df.assign(order=lambda d: d.category.map(order))
        .sort_values(["year", "order"])
        .drop("order", axis=1)
        .loc[lambda d: d.year < 2021]
    )
    df.to_csv(
        config.Paths.output / "debt_health_2020.csv",
        index=False,
    )
