import pandas as pd

from scripts.config import Paths
from bblocks import add_iso_codes_column, set_bblocks_data_path, DebtIDS

set_bblocks_data_path(Paths.raw_data)


def update_debt_service(star_year: int, end_year: int) -> None:
    """Update the debt service data"""
    ids = DebtIDS()

    service_indicators = ids.debt_service_indicators()

    ids.load_data(
        indicators=list(service_indicators), start_year=star_year, end_year=end_year
    )

    df = ids.get_data()

    df.to_feather(Paths.raw_data / "ids_service_raw.feather")


def read_debt_service() -> pd.DataFrame:
    """Read the debt service data"""
    return pd.read_feather(Paths.raw_data / "ids_service_raw.feather")


def _filter_world_counterpart(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the data to only include the world as a counterpart"""
    return df.query("counterpart_area == 'World'").reset_index(drop=True)


def _create_service_total(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate into a total debt service for each year and country"""
    return df.groupby(["country", "year"], as_index=False)["value"].sum()


def _filter_year(df: pd.DataFrame, year: int = 2020) -> pd.DataFrame:
    """Filter the data to only include the world as a counterpart"""
    return (
        df.query(f"year.dt.year == {year}")
        .reset_index(drop=True)
        .assign(year=lambda d: d.year.dt.year)
    )


def _keep_valid_iso(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only if ISO column has 3 characters"""

    return df.query("iso_code.str.len() == 3").reset_index(drop=True)


def service_data() -> pd.DataFrame:
    return (
        read_debt_service()
        .pipe(_filter_world_counterpart)
        .pipe(_create_service_total)
        .pipe(add_iso_codes_column, id_column="country", id_type="regex")
        .pipe(_keep_valid_iso)
    )
