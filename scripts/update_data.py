"""Update data for the project"""

from scripts.logger import logger
from scripts.debt.debt_service import update_debt_service


def update_data() -> None:
    """Pipeline to update data"""

    update_debt_service(star_year=2000, end_year=2021)
    logger.info("Updated debt service data")

    #TODO: Add other data updates here

    logger.info("Successfully updated all data")


if __name__ == "__main__":
    update_data()
