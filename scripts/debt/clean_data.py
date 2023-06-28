import pandas as pd
from bblocks import DebtIDS, add_income_level_column, convert_id


def _clean_counterpart_area(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the non-breaking space from the counterpart_area column
    and harmomise the names of the counterpart areas (when possible)."""
    return df.assign(
        counterpart_area=lambda d: d.counterpart_area.apply(
            lambda r: r.replace("\xa0", "")
        )
    ).assign(
        counterpart_area=lambda d: convert_id(
            d["counterpart_area"], from_type="regex", to_type="name_short"
        )
    )


def _year2int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the year column to int"""
    return df.assign(year=lambda d: d.year.dt.year)


def _add_continent(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        continent=lambda d: convert_id(
            d.country, from_type="regex", to_type="continent"
        )
    )


def _clean_indicators(
    df: pd.DataFrame,
    filter_counterparts: bool = True,
    filter_columns: bool = True,
    counterparts: list = None,
) -> pd.DataFrame:
    """Clean the IDS data.

    Optionally filter counterparts to keep only those in study_counterparts.
    Optionally filter columns to keep only the ones needed for the analysis.

    """

    df = (
        df.pipe(_clean_counterpart_area)
        .pipe(add_income_level_column, id_column="country", id_type="regex")
        .pipe(_add_continent)
        .pipe(_year2int)
        .dropna(subset=["income_level"])
    )

    if filter_counterparts:
        df = df.query(f"counterpart_area in {counterparts}")

    if filter_columns:
        df = df.filter(
            [
                "country",
                "counterpart_area",
                "income_level",
                "continent",
                "series_code",
                "year",
                "value",
            ],
            axis=1,
        )

    return df.reset_index(drop=True)


def get_clean_data(
    start_year,
    end_year,
    indicators: list | dict | str,
    filter_counterparts: bool = False,
    counterparts: list | dict = None,
) -> pd.DataFrame:
    """Get indicator data for each country/counterpart_area pair."""

    if counterparts is None and filter_counterparts:
        raise ValueError(
            "counterparts must be specified if filter_counterparts is True"
        )

    # Create IDS object
    ids = DebtIDS()

    # Load data
    ids.load_data(indicators=indicators, start_year=start_year, end_year=end_year)

    # Get data and clean it
    df = ids.get_data().pipe(
        _clean_indicators,
        filter_counterparts=filter_counterparts,
        counterparts=list(counterparts),
    )

    if filter_counterparts:
        # Make sure only the right indicators are kept for each counterpart
        if isinstance(indicators, dict):
            condition = ""
            indicator_types = {v: k for k, v in indicators.items()}
            for counterpart, indicator_type in counterparts.items():
                condition += (
                    f"(counterpart_area == '{counterpart}' and "
                    f"series_code == '{indicator_types[indicator_type]}') or "
                )

            df = df.query(condition[:-4]).reset_index(drop=True)

    return df.drop(columns=["series_code"])
