import pandas as pd

from customer_churn_prediction.logger import DatabaseError, get_logger

from .columns import FEATURE_COLUMNS, RAW_COLUMNS
from .connection import get_connection, get_engine

# Create logger
logger = get_logger("db_operations", "database.log")


class DatabaseOperations:
    """Handles database operations for churn_data and predictions."""

    def __init__(self):
        """Initialize DB connection and cursor."""

        self.conn = get_connection()
        self.engine = get_engine()
        self.cur = self.conn.cursor()

    def insert_dataframe(self, df):
        """Insert raw dataframe rows into churn_data table."""

        try:
            df = df[RAW_COLUMNS]

            # Convert NaN values to None
            df = df.astype(object).where(df.notna(), None)

            query = f"""
                INSERT INTO churn_data (
                    {", ".join(RAW_COLUMNS)}
                )
                VALUES (
                    {", ".join(["%s"] * len(RAW_COLUMNS))}
                )
            """

            data = [tuple(row) for row in df.itertuples(index=False, name=None)]

            self.cur.executemany(query, data)
            self.conn.commit()

            logger.info(f"Inserted {len(df)} rows into churn_data.")

        except Exception as e:
            self.conn.rollback()

            logger.error(
                f"Failed to insert rows - {len(df)}: {e}",
                exc_info=True,
            )
            raise DatabaseError("Failed to insert dataframe") from e

    def fetch_data(self):
        """Fetch all data from churn_data table."""

        try:
            return pd.read_sql(
                "SELECT * FROM churn_data",
                self.engine,
            )

        except Exception as e:
            logger.error(
                f"Failed to fetch data from churn_data: {e}",
                exc_info=True,
            )
            raise DatabaseError("Failed to fetch data") from e

    def clear_table(self):
        """Delete all data from churn_data table."""

        try:
            self.cur.execute("DELETE FROM churn_data")

            self.conn.commit()

            logger.info("Table churn_data cleared.")

        except Exception as e:
            self.conn.rollback()

            logger.error(
                f"Failed to clear churn_data: {e}",
                exc_info=True,
            )
            raise DatabaseError("Failed to clear table") from e

    def save_prediction(
        self,
        input_data,
        prediction,
        probability,
    ):
        """Insert a prediction result into predictions table."""

        try:
            query = f"""
                INSERT INTO predictions (
                    {", ".join(FEATURE_COLUMNS)},
                    predicted_churn,
                    churn_probability
                )
                VALUES (
                    {", ".join(["%s"] * (len(FEATURE_COLUMNS) + 2))}
                )
            """

            self.cur.execute(
                query,
                (
                    *input_data,
                    bool(prediction),
                    float(probability),
                ),
            )

            self.conn.commit()

            logger.info("Prediction saved successfully.")

        except Exception as e:
            self.conn.rollback()

            logger.error(
                f"Failed to save prediction: {e}",
                exc_info=True,
            )
            raise DatabaseError("Failed to save prediction") from e

    def fetch_predictions(self):
        """Fetch all data from predictions table."""

        try:
            return pd.read_sql(
                "SELECT * FROM predictions",
                self.engine,
            )

        except Exception as e:
            logger.error(
                f"Failed to fetch predictions: {e}",
                exc_info=True,
            )
            raise DatabaseError("Failed to fetch predictions") from e

    def close_connection(self):
        """Close database connection and engine."""

        try:
            self.cur.close()
            self.conn.close()
            self.engine.dispose()

            logger.info("Database connection closed.")

        except Exception as e:
            logger.error(
                f"Failed to close database connection: {e}",
                exc_info=True,
            )
            raise DatabaseError("Failed to close database connection") from e
