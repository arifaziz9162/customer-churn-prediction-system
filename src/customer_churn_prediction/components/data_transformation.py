import pandas as pd
from sklearn.model_selection import train_test_split

from customer_churn_prediction.entity.config_entity import DataTransformationConfig
from customer_churn_prediction.logger import DataTransformationError, get_logger
from customer_churn_prediction.utils.common import save_json, save_status

logger = get_logger("data_transformation", "data_transformation.log")


class DataTransformation:
    """Clean, split, and save the training data."""

    def __init__(self, config: DataTransformationConfig):
        """Initialize data transformation with configuration."""

        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load input dataset from CSV file."""

        try:
            file_path = self.config.raw_data_path
            logger.info(f"Reading file: '{file_path}'")
            df = pd.read_csv(file_path)
            df = df.rename(
                columns={
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
            )
            df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
            logger.info(f"Data loaded successfully - shape: '{df.shape}'")
            logger.info(f"Columns found: '{df.columns.tolist()}'")

            return df

        except Exception as e:
            logger.error(
                f"Load data failed - path='{self.config.raw_data_path}': {e}",
                exc_info=True,
            )
            raise DataTransformationError("Failed to load data") from e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ID column and convert the target column to numeric values."""

        try:
            df = df.copy()

            # Remove customer ID because it is not useful for prediction
            df = df.drop(columns=[self.config.id_column])

            target = self.config.target_column

            # Convert target values using configured mapping
            df[target] = df[target].map(self.config.target_mapping)

            if df[target].isnull().any():
                logger.error(f"Target column contains unmapped values: '{target}'")
                raise DataTransformationError("Target column contains invalid values")

            df[target] = df[target].astype(int)

            logger.info(f"Data cleaning completed - shape: '{df.shape}'")
            logger.info(f"Target column '{target}' converted successfully")

            return df

        except DataTransformationError:
            raise

        except Exception as e:
            logger.error(
                f"Data cleaning failed: {e}",
                exc_info=True,
            )
            raise DataTransformationError("Failed to clean data") from e

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataset into stratified train and test sets."""

        try:
            test_size = self.config.test_size
            random_state = self.config.random_state
            target = self.config.target_column

            train, test = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[target],
            )

            logger.info(f"Train dataset shape: '{train.shape}'")
            logger.info(f"Test dataset shape: '{test.shape}'")

            logger.info(f"Train churn rate: '{train[target].mean():.4f}'")
            logger.info(f"Test churn rate: '{test[target].mean():.4f}'")

            return train, test

        except Exception as e:
            logger.error(
                f"Data split failed: {e}",
                exc_info=True,
            )
            raise DataTransformationError("Failed to split data") from e

    def save_data(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ):
        """Save train and test datasets to CSV files."""

        try:
            train.to_csv(
                self.config.train_data_path,
                index=False,
            )

            test.to_csv(
                self.config.test_data_path,
                index=False,
            )

            logger.info(f"Train file saved: '{self.config.train_data_path}'")
            logger.info(f"Test file saved: '{self.config.test_data_path}'")

        except Exception as e:
            logger.error(
                f"Data save failed: {e}",
                exc_info=True,
            )
            raise DataTransformationError("Failed to save transformed data") from e

    def run(self):
        """Run the data transformation process."""

        try:
            df = self.load_data()
            df = self.clean_data(df)
            train, test = self.split_data(df)
            self.save_data(train, test)

            save_json(
                self.config.status_file,
                {"status": True, "stage": "data_transformation"},
            )

            save_status(self.config.status_file, "data_transformation", True)
            logger.info("Data transformation completed successfully")

        except Exception as e:
            save_json(
                self.config.status_file,
                {"status": False, "stage": "data_transformation"},
            )

            try:
                save_status(
                    self.config.status_file, "data_transformation", False, str(e)
                )
            except Exception:
                logger.exception("Failed to save data transformation status")
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise DataTransformationError("Data transformation process failed") from e
