from scripts import config
from scripts.logger import logger

import pandas as pd
import requests


def get_fed_data(vintage: str | None = None) -> pd.DataFrame:
    if vintage is None:
        vintage = pd.Timestamp.today().strftime("%Y-%m-%d")

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        "id=FEDFUNDS&"
        f"vintage_date={vintage}&revision_date={vintage}&nd=1954-07-01"
    )

    try:
        return pd.read_csv(url, parse_dates=["DATE"]).rename(
            columns={"FEDFUNDS": "effective_rate", "DATE": "date"}
        )

    except requests.exceptions.HTTPError as e:
        logger.info(f"Error downloading data: {e}")

        # vintage is yesterday
        vintage = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        return get_fed_data(vintage)


def hike_periods() -> dict[str, tuple[str, str]]:
    return {
        "'87-'89": ("1986-10-01", "1989-04-01"),
        "'94-'95": ("1994-01-01", "1995-04-01"),
        "'99-'00": ("1999-01-01", "2000-07-01"),
        "'04-'06": ("2004-05-01", "2006-08-01"),
        "'15-'18": ("2015-11-01", "2019-01-01"),
        "'22-?": ("2022-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")),
    }


def base_hike_start(df: pd.DataFrame, hikes: dict) -> pd.DataFrame:
    """Calculate the change in rate from the start of the rate hike cycle.

    Adds a column with the number of months since the start of the rate hike cycle.
    Adds a column with the cycle name.
    """
    hikes_dfs = pd.DataFrame()
    for name, (start, end) in hikes.items():
        d_ = df.query(f"date >= '{start}' & date <= '{end}'").assign(
            change=lambda d: round(d.effective_rate - d.effective_rate.min(), 4),
            months=lambda d: (
                (d.date.dt.year - d.date.min().date().year) * 12
                + df.date.dt.month
                - d.date.min().date().month
            ).astype("Int32"),
            cycle=name,
        )
        hikes_dfs = pd.concat([hikes_dfs, d_])

    return hikes_dfs


def reformat_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(date=lambda d: d.date.dt.strftime("%B %Y"))


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.filter(["change", "months", "cycle", "date", "effective_rate"], axis=1)


def update_fed_rate_hikes_chart_data() -> None:
    """Pipeline to get, clean, and process the data for the Fed Rate Hikes chart."""
    df = get_fed_data()
    hikes = hike_periods()

    hikes_data = base_hike_start(df, hikes).pipe(reformat_data).pipe(filter_columns)

    hikes_data.to_csv(config.Paths.output / "fed_rate_hikes.csv", index=False)


def wide_fed_rates_chart() -> None:
    df = pd.read_csv(config.Paths.output / "fed_rate_hikes.csv")
    df2 = df.pivot(
        index=["months", "date", "effective_rate"], columns="cycle", values="change"
    )


if __name__ == "__main__":
    update_fed_rate_hikes_chart_data()
