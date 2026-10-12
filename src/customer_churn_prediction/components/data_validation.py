import pandas as pd
from pandas.api import types as pdt

from customer_churn_prediction.entity.config_entity import DataValidationConfig
from customer_churn_prediction.logger import DataValidationError, get_logger
from customer_churn_prediction.utils.common import save_json, save_status

COLUMN_MAPPING = {
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


logger = get_logger("data_validation", "data_validation.log")


class DataValidation:
    """Handles dataset validation checks."""

    def __init__(self, config: DataValidationConfig):
        """Initialize data validation with configuration."""

        self.config = config

    @staticmethod
    def _dtype_matches(series: pd.Series, expected_dtype: str) -> bool:
        """Check whether the column has the expected data type."""

        if expected_dtype == "int":
            return pdt.is_integer_dtype(series)
        if expected_dtype == "float":
            return pdt.is_float_dtype(series) or pdt.is_integer_dtype(series)
        if expected_dtype == "str":
            return pdt.is_object_dtype(series) or pdt.is_string_dtype(series)

        return False

    def load_data(self) -> pd.DataFrame:
        try:
            file_path = self.config.raw_data_path
            logger.info(f"Reading file: '{file_path}'")
            df = pd.read_csv(file_path)
            df = df.rename(columns=COLUMN_MAPPING)
            if "total_charges" in df.columns:
                df["total_charges"] = pd.to_numeric(
                    df["total_charges"], errors="coerce"
                )
            logger.info(f"Data loaded successfully with shape: '{df.shape}'")
            logger.info(f"Columns found: '{df.columns.tolist()}'")
            return df

        except Exception as e:
            logger.error(
                f"Load data failed - path='{self.config.raw_data_path}': {e}",
                exc_info=True,
            )
            raise DataValidationError("Failed to load data") from e

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Check that expected columns are present and no unexpected columns exist."""

        try:
            expected_columns = list(self.config.columns)

            missing_columns = [
                column for column in expected_columns if column not in df.columns
            ]

            unexpected_columns = [
                column for column in df.columns if column not in expected_columns
            ]

            if missing_columns:
                logger.error(f"Missing columns found: '{missing_columns}'")

            if unexpected_columns:
                logger.error(f"Unexpected columns found: '{unexpected_columns}'")

            if missing_columns or unexpected_columns:
                return False

            logger.info("Column validation passed.")
            return True

        except Exception as e:
            logger.error(f"Column validation error: {e}", exc_info=True)
            raise DataValidationError("Column validation failed") from e

    def validate_dtypes(self, df: pd.DataFrame) -> bool:
        """Check data types of expected columns."""

        try:
            all_passed = True

            for column, expected_dtype in self.config.columns.items():
                if column not in df.columns:
                    logger.error(
                        f"Dtype validation failed - missing column: '{column}'"
                    )
                    all_passed = False
                    continue

                if not self._dtype_matches(df[column], expected_dtype):
                    logger.error(
                        f"Dtype mismatch - column='{column}' "
                        f"expected='{expected_dtype}' "
                        f"got='{df[column].dtype}'"
                    )
                    all_passed = False

            if all_passed:
                logger.info("Data type validation passed.")

            return all_passed

        except Exception as e:
            logger.error(
                f"Data type validation error: {e}",
                exc_info=True,
            )
            raise DataValidationError("Data type validation failed") from e

    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        """Check that non-nullable columns do not contain missing values."""

        try:
            all_passed = True

            for column in df.columns:
                null_count = int(df[column].isnull().sum())

                if null_count > 0 and column not in self.config.nullable_columns:
                    logger.error(
                        f"Missing values found - column='{column}' count='{null_count}'"
                    )
                    all_passed = False

            if all_passed:
                logger.info("Missing value validation passed.")

            return all_passed

        except Exception as e:
            logger.error(f"Missing value validation error: {e}", exc_info=True)
            raise DataValidationError("Missing value validation failed") from e

    def validate_categorical_values(self, df: pd.DataFrame) -> bool:
        """Check that categorical columns contain allowed values."""

        try:
            all_passed = True

            allowed_values = dict(self.config.categorical_values)

            allowed_values[self.config.target_column] = self.config.target_values

            for column, expected_values in allowed_values.items():
                if column not in df.columns:
                    continue

                actual_values = set(df[column].dropna().unique())
                invalid_values = sorted(actual_values - set(expected_values))

                if invalid_values:
                    logger.error(
                        f"Invalid categorical values - "
                        f"column='{column}' "
                        f"values='{invalid_values}'"
                    )
                    all_passed = False

            if all_passed:
                logger.info("Categorical value validation passed.")

            return all_passed

        except Exception as e:
            logger.error(f"Categorical value validation error: {e}", exc_info=True)
            raise DataValidationError("Categorical value validation failed") from e

    def validate_numerical_ranges(self, df: pd.DataFrame) -> bool:
        """Check numerical columns for invalid negative values."""

        try:
            all_passed = True

            for column in self.config.numerical_columns:
                if column not in df.columns:
                    continue

                if not pdt.is_numeric_dtype(df[column]):
                    continue

                negative_count = int((df[column] < 0).sum())

                if negative_count > 0:
                    logger.error(
                        f"Negative values found - "
                        f"column='{column}' "
                        f"count='{negative_count}'"
                    )
                    all_passed = False

            if all_passed:
                logger.info("Numerical range validation passed.")

            return all_passed

        except Exception as e:
            logger.error(f"Numerical range validation error: {e}", exc_info=True)
            raise DataValidationError("Numerical range validation failed") from e

    def validate_unique_ids(self, df: pd.DataFrame) -> bool:
        """Check that customer IDs are unique."""

        try:
            column = self.config.id_column

            if column not in df.columns:
                logger.error(
                    f"Unique ID validation failed - missing column: '{column}'"
                )
                return False

            duplicate_count = int(df[column].duplicated().sum())

            if duplicate_count > 0:
                logger.error(
                    f"Duplicate IDs found - "
                    f"column='{column}' "
                    f"duplicates='{duplicate_count}'"
                )
                return False

            logger.info("Unique ID validation passed.")
            return True

        except Exception as e:
            logger.error(f"Unique ID validation error: {e}", exc_info=True)
            raise DataValidationError("Unique ID validation failed") from e

    def run(self):
        """Run the data validation process."""

        try:
            df = self.load_data()

            column_check = self.validate_columns(df)
            dtype_check = self.validate_dtypes(df)
            missing_check = self.validate_missing_values(df)
            categorical_check = self.validate_categorical_values(df)
            range_check = self.validate_numerical_ranges(df)
            unique_id_check = self.validate_unique_ids(df)

            overall_status = all(
                [
                    column_check,
                    dtype_check,
                    missing_check,
                    categorical_check,
                    range_check,
                    unique_id_check,
                ]
            )

            report = {
                "status": "passed" if overall_status else "failed",
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_values": {
                    column: int(count)
                    for column, count in df.isnull().sum().items()
                    if count > 0
                },
                "checks": {
                    "columns": column_check,
                    "dtypes": dtype_check,
                    "missing_values": missing_check,
                    "categorical_values": categorical_check,
                    "numerical_ranges": range_check,
                    "unique_ids": unique_id_check,
                },
            }

            save_json(self.config.report_path, report)
            save_status(self.config.status_file, "data_validation", True)

            if overall_status:
                logger.info("Data validation completed successfully.")
            else:
                logger.error("Data validation failed.")

        except Exception as e:
            try:
                save_status(self.config.status_file, "data_validation", False, str(e))
            except Exception:
                logger.exception("Failed to save data validation status")
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise DataValidationError("Data validation process failed") from e

        if not overall_status:
            raise DataValidationError(
                "Validation checks failed, stopping the pipeline here"
            )

        return report
