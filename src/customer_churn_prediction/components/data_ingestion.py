import os
import shutil

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from customer_churn_prediction.database import RAW_COLUMNS, DatabaseOperations
from customer_churn_prediction.entity import DataIngestionConfig
from customer_churn_prediction.logger import DataIngestionError, get_logger
from customer_churn_prediction.utils import get_size, save_status

load_dotenv()

logger = get_logger("data_ingestion", "data_ingestion.log")


class DataIngestion:
    """Download the customer churn dataset and store it in PostgreSQL."""

    def __init__(self, config: DataIngestionConfig):
        """Initialize data ingestion with configuration."""
        self.config = config

    def download_file(self) -> None:
        """Download the configured dataset from Kaggle."""

        file_path = self.config.raw_data_path

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.download_dir.mkdir(parents=True, exist_ok=True)

            if file_path.is_file():
                logger.info("Dataset already exists at '%s'.", file_path)
                return

            kaggle_username = os.getenv("KAGGLE_USERNAME")
            kaggle_api_token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_KEY")

            if not kaggle_username or not kaggle_api_token:
                raise ValueError(
                    "KAGGLE_USERNAME and KAGGLE_API_TOKEN or KAGGLE_KEY must be configured."
                )

            logger.info("Downloading Kaggle dataset '%s'.", self.config.kaggle_dataset)

            api = KaggleApi()
            api.authenticate()

            api.dataset_download_files(
                dataset=self.config.kaggle_dataset,
                path=str(self.config.download_dir),
                unzip=True,
            )

            downloaded_file = self.config.download_dir / self.config.kaggle_file_name

            if not downloaded_file.is_file():
                available_files = [
                    path.name for path in self.config.download_dir.glob("*.csv")
                ]
                raise FileNotFoundError(
                    f"Expected file '{downloaded_file}' was not found. Available CSV files: {available_files}"
                )

            shutil.copy2(downloaded_file, file_path)

            logger.info("Dataset downloaded successfully to '%s'.", file_path)
            logger.info("Dataset file size: %s", get_size(str(file_path)))

        except Exception as e:
            logger.error(
                "Dataset download failed for '%s': %s", file_path, e, exc_info=True
            )
            raise DataIngestionError(
                "Failed to download the customer churn dataset."
            ) from e

    def load_csv(self) -> pd.DataFrame:
        """Load the raw CSV and normalize column names."""
        file_path = self.config.raw_data_path
        try:
            if not file_path.is_file():
                raise FileNotFoundError(f"Dataset not found: '{file_path}'")

            df = pd.read_csv(file_path)

            column_mapping = {
                "customerID": "customer_id",
                "SeniorCitizen": "senior_citizen",
                "Partner": "partner",
                "Dependents": "dependents",
                "PhoneService": "phone_service",
                "MultipleLines": "multiple_lines",
                "InternetService": "internet_service",
                "OnlineSecurity": "online_security",
                "OnlineBackup": "online_backup",
                "DeviceProtection": "device_protection",
                "TechSupport": "tech_support",
                "StreamingTV": "streaming_tv",
                "StreamingMovies": "streaming_movies",
                "Contract": "contract",
                "PaperlessBilling": "paperless_billing",
                "PaymentMethod": "payment_method",
                "MonthlyCharges": "monthly_charges",
                "TotalCharges": "total_charges",
                "Churn": "churn",
            }

            df = df.rename(columns=column_mapping)

            missing_columns = [
                column for column in RAW_COLUMNS if column not in df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Dataset is missing required columns: {missing_columns}"
                )

            df = df[RAW_COLUMNS].copy()
            df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
            df["tenure"] = pd.to_numeric(df["tenure"], errors="raise")
            df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="raise")

            logger.info("CSV loaded successfully. Shape: %s", df.shape)
            logger.info("Normalized columns: %s", df.columns.tolist())
            logger.info("Missing values by column: %s", df.isna().sum().to_dict())

            return df

        except Exception as e:
            logger.error("Failed to load CSV '%s': %s", file_path, e, exc_info=True)
            raise DataIngestionError("Failed to load and validate the raw CSV.") from e

    def store_in_database(self, df: pd.DataFrame) -> None:
        """Store raw data in PostgreSQL if the table is empty."""

        db = None

        try:
            db = DatabaseOperations()
            existing = db.fetch_data()

            if not existing.empty:
                logger.info(
                    "Table 'churn_data' already contains data. Skipping insertion."
                )
                return

            logger.info("Inserting %s rows into 'churn_data'.", len(df))
            db.insert_dataframe(df)
            logger.info("Data inserted successfully into PostgreSQL.")

        except Exception as e:
            logger.error(
                "Database insertion failed for %s rows: %s", len(df), e, exc_info=True
            )
            raise DataIngestionError(
                "Failed to store the dataset in PostgreSQL."
            ) from e

        finally:
            if db is not None:
                try:
                    db.close_connection()
                except Exception:
                    logger.exception("Failed to close the database connection.")

    def run(self) -> None:
        """Execute the complete data ingestion stage."""

        try:
            self.download_file()
            df = self.load_csv()
            self.store_in_database(df)
            save_status(self.config.status_file, "data_ingestion", True)
            logger.info("Data ingestion completed successfully.")

        except Exception as e:
            try:
                save_status(self.config.status_file, "data_ingestion", False, str(e))
            except Exception:
                logger.exception("Failed to save data ingestion status")
            logger.exception("Failed to save data ingestion failure status.")
            raise DataIngestionError("Data ingestion process failed.") from e
