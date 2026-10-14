"""Serving configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Configuration for the customer churn serving application."""

    SERVICE_NAME: str = "customer_churn_service"

    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

    # Exact BentoML model tag can be supplied through the environment.
    BENTO_MODEL_TAG: str = os.getenv(
        "BENTO_MODEL_TAG",
        "customer_churn_model:kmcgjpwfiwitgaav",
    )

    MODEL_FILE_NAME: str = "model.joblib"

    # API traffic settings.
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "32"))

    # Classification settings.
    CLASSIFICATION_THRESHOLD: float = float(
        os.getenv("CLASSIFICATION_THRESHOLD", "0.3")
    )

    # Prediction persistence settings.
    DB_SAVE_ENABLED: bool = os.getenv("DB_SAVE_ENABLED", "true").strip().lower() in {
        "true",
        "1",
        "yes",
    }
    DB_SAVE_WORKERS: int = int(os.getenv("DB_SAVE_WORKERS", "2"))
    PREDICTIONS_MAX_LIMIT: int = 500

    def validate(self) -> None:
        """Validate configuration before starting the service."""

        if not self.BENTO_MODEL_TAG.strip():
            raise ValueError("BENTO_MODEL_TAG must not be empty.")

        if not 0.0 < self.CLASSIFICATION_THRESHOLD < 1.0:
            raise ValueError("CLASSIFICATION_THRESHOLD must be between 0 and 1.")

        if self.REQUEST_TIMEOUT_SECONDS <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be greater than zero.")

        if self.MAX_CONCURRENCY <= 0:
            raise ValueError("MAX_CONCURRENCY must be greater than zero.")

        if self.DB_SAVE_WORKERS <= 0:
            raise ValueError("DB_SAVE_WORKERS must be greater than zero.")


settings = Settings()
settings.validate()
