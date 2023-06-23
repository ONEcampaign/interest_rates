import pandas as pd
from bblocks import add_short_names_column

from scripts import config
from scripts.debt.debt_service import service_data
from scripts.government.revenue import get_gdp_usd, get_government_expenditure_gdp


def debt_gdp() -> pd.DataFrame:
    debt = service_data()
    gdp = get_gdp_usd()
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


def debt_exp() -> pd.DataFrame:
    debt = debt_gdp()

    return _gdp2exp(debt, "Debt (% Expenditure)")


def health_spending() -> pd.DataFrame:
    health = pd.read_csv(config.Paths.raw_data / "health_spending_gdp.csv")

    return _gdp2exp(health, "Health (% Expenditure)")


def education_spending() -> pd.DataFrame:
    education = pd.read_csv(config.Paths.raw_data / "education_spending_gdp.csv")

    return _gdp2exp(education, "Education (% Expenditure)")


def debt_education_health_comparison_chart() -> pd.DataFrame:
    debt = debt_exp()
    health = health_spending()
    education = education_spending()

    df = (
        pd.merge(debt, health, on=["iso_code", "year"], suffixes=("_debt", "_health"))
        .merge(
            education.rename(columns={"value": "value_education"}),
            on=["iso_code", "year"],
        )
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
