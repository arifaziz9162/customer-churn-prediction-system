"""Unit tests for customer churn data ingestion."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from customer_churn_prediction.components.data_ingestion import (
    DataIngestion,
)
from customer_churn_prediction.logger import DataIngestionError


@pytest.fixture
def config(tmp_path):
    """Create a temporary data ingestion configuration."""

    config = MagicMock()
    config.raw_data_path = tmp_path / "raw" / "customer_churn.csv"
    config.download_dir = tmp_path / "downloads"
    config.kaggle_dataset = "test/customer-churn"
    config.kaggle_file_name = "customer_churn.csv"
    config.status_file = tmp_path / "status.json"

    return config


@pytest.fixture
def ingestion(config):
    """Create a data ingestion instance."""

    return DataIngestion(config)


@pytest.fixture
def raw_dataframe():
    """Create a minimal dataset with the original Kaggle column names."""

    return pd.DataFrame(
        {
            "customerID": ["C001", "C002"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [12, 24],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["Yes", "No"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["Yes", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
            ],
            "MonthlyCharges": [50.5, 70.0],
            "TotalCharges": ["606.0", "1680.0"],
            "Churn": ["No", "Yes"],
        }
    )


def test_download_file_skips_existing_file(ingestion):
    """Do not download the dataset if it already exists."""

    ingestion.config.raw_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ingestion.config.raw_data_path.write_text("existing data")

    with patch(
        "customer_churn_prediction.components.data_ingestion.KaggleApi"
    ) as kaggle_api:
        ingestion.download_file()

    kaggle_api.assert_not_called()


def test_download_file_requires_credentials(ingestion):
    """Raise a domain exception when Kaggle credentials are missing."""

    with (
        patch(
            "customer_churn_prediction.components.data_ingestion.os.getenv",
            return_value=None,
        ),
        pytest.raises(DataIngestionError),
    ):
        ingestion.download_file()


def test_download_file_copies_dataset(ingestion, tmp_path):
    """Download and copy the expected dataset into the raw directory."""

    download_dir = ingestion.config.download_dir
    download_dir.mkdir(parents=True)

    expected_file = download_dir / ingestion.config.kaggle_file_name
    expected_file.write_text("customerID,Churn\nC001,No\n")

    mock_api = MagicMock()

    with (
        patch(
            "customer_churn_prediction.components.data_ingestion.os.getenv",
            side_effect=lambda key: {
                "KAGGLE_USERNAME": "test-user",
                "KAGGLE_API_TOKEN": "test-token",
            }.get(key),
        ),
        patch(
            "customer_churn_prediction.components.data_ingestion.KaggleApi",
            return_value=mock_api,
        ),
        patch(
            "customer_churn_prediction.components.data_ingestion.get_size",
            return_value="100 bytes",
        ),
    ):
        ingestion.download_file()

    assert ingestion.config.raw_data_path.is_file()
    assert ingestion.config.raw_data_path.read_text() == expected_file.read_text()

    mock_api.authenticate.assert_called_once()
    mock_api.dataset_download_files.assert_called_once()


def test_load_csv_normalizes_columns(
    ingestion,
    raw_dataframe,
):
    """Normalize Kaggle columns and preserve the configured column order."""

    ingestion.config.raw_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dataframe.to_csv(
        ingestion.config.raw_data_path,
        index=False,
    )

    result = ingestion.load_csv()

    assert "customer_id" in result.columns
    assert "monthly_charges" in result.columns
    assert "total_charges" in result.columns
    assert "churn" in result.columns

    assert "customerID" not in result.columns
    assert pd.api.types.is_numeric_dtype(result["total_charges"])

    from customer_churn_prediction.database import RAW_COLUMNS

    assert result.columns.tolist() == RAW_COLUMNS


def test_load_csv_raises_when_file_missing(ingestion):
    """Raise DataIngestionError if the raw CSV is missing."""

    with pytest.raises(DataIngestionError):
        ingestion.load_csv()


def test_load_csv_raises_for_missing_required_columns(
    ingestion,
):
    """Reject a CSV missing required dataset columns."""

    ingestion.config.raw_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "customerID": ["C001"],
            "Churn": ["No"],
        }
    ).to_csv(
        ingestion.config.raw_data_path,
        index=False,
    )

    with pytest.raises(DataIngestionError):
        ingestion.load_csv()


def test_store_in_database_skips_existing_data(
    ingestion,
    raw_dataframe,
):
    """Avoid inserting duplicate source data into a populated table."""

    normalized = raw_dataframe.rename(
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

    db = MagicMock()
    db.fetch_data.return_value = pd.DataFrame({"customer_id": ["EXISTING"]})

    with patch(
        "customer_churn_prediction.components.data_ingestion.DatabaseOperations",
        return_value=db,
    ):
        ingestion.store_in_database(normalized)

    db.insert_dataframe.assert_not_called()
    db.close_connection.assert_called_once()


def test_store_in_database_inserts_when_empty(
    ingestion,
    raw_dataframe,
):
    """Insert source data when the database table is empty."""

    db = MagicMock()
    db.fetch_data.return_value = pd.DataFrame()

    with patch(
        "customer_churn_prediction.components.data_ingestion.DatabaseOperations",
        return_value=db,
    ):
        ingestion.store_in_database(raw_dataframe)

    db.insert_dataframe.assert_called_once_with(raw_dataframe)
    db.close_connection.assert_called_once()
