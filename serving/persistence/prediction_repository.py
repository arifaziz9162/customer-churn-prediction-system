"""Repository for storing and retrieving customer churn predictions."""

import json
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from customer_churn_prediction.database import DatabaseOperations
from customer_churn_prediction.logger import (
    DatabaseError,
    get_logger,
)
from serving.config import settings
from serving.metrics import DB_SAVE_TOTAL

logger = get_logger("prediction_repository", "serving.log")


class PredictionRepository:
    """Persist prediction history using background database workers."""

    def __init__(self, workers: int | None = None) -> None:
        """Initialize the database write executor."""

        self._executor = ThreadPoolExecutor(
            max_workers=workers or settings.DB_SAVE_WORKERS,
            thread_name_prefix="prediction-db",
        )

    @staticmethod
    def _to_python_value(value: Any) -> Any:
        """Convert NumPy scalar values into native Python values."""

        if hasattr(value, "item"):
            return value.item()

        return value

    def save(
        self,
        features: list[Any],
        prediction: int,
        probability: float,
    ) -> None:
        """Save a prediction using native Python values."""

        db = None

        try:
            # Convert NumPy scalar values before sending data to PostgreSQL.
            clean_features = [self._to_python_value(value) for value in features]

            clean_prediction = int(self._to_python_value(prediction))

            clean_probability = float(self._to_python_value(probability))

            db = DatabaseOperations()

            db.save_prediction(
                clean_features,
                clean_prediction,
                clean_probability,
            )

        except Exception:
            logger.exception("Failed to persist customer churn prediction.")
            raise

        finally:
            if db is not None:
                db.close_connection()

    def save_async(
        self,
        features: list[Any],
        prediction: int,
        probability: float,
    ) -> Future:
        """Schedule a prediction database write in the background."""
        future = self._executor.submit(
            self.save,
            features,
            prediction,
            probability,
        )

        future.add_done_callback(self._on_saved)
        return future

    @staticmethod
    def _on_saved(future: Future) -> None:
        """Record the outcome of a background database write."""

        if future.cancelled():
            DB_SAVE_TOTAL.labels(status="failure").inc()
            logger.warning("Prediction database write was cancelled.")
            return

        error = future.exception()

        if error is None:
            DB_SAVE_TOTAL.labels(status="success").inc()
            logger.info("Prediction saved to PostgreSQL successfully.")
            return

        DB_SAVE_TOTAL.labels(status="failure").inc()

        logger.error(
            "Prediction database write failed: %s",
            error,
            exc_info=(
                type(error),
                error,
                error.__traceback__,
            ),
        )

    def fetch_recent(self, limit: int) -> list[dict[str, Any]]:
        """Fetch recent prediction records from the database."""

        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        db = None

        try:
            db = DatabaseOperations()
            frame = db.fetch_predictions()

            if frame.empty:
                return []

            frame = frame.tail(limit)

            return json.loads(
                frame.to_json(
                    orient="records",
                    date_format="iso",
                )
            )

        except DatabaseError:
            logger.exception("Failed to fetch prediction history.")
            raise

        finally:
            if db is not None:
                db.close_connection()

    def close(self) -> None:
        """Wait for pending database writes before shutting down."""
        self._executor.shutdown(wait=True)
