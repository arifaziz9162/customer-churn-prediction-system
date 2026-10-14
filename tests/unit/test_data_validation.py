"""Unit tests for customer churn data validation."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from customer_churn_prediction.components.data_validation import (
    DataValidation,
)
from customer_churn_prediction.logger import DataValidationError


@pytest.fixture
def config():
    """Create a validation configuration."""

    config = MagicMock()
    config.raw_data_path = "unused.csv"

    config.columns = {
        "customer_id": "str",
        "tenure": "int",
        "monthly_charges": "float",
        "total_charges": "float",
        "churn": "str",
    }

    config.nullable_columns = ["total_charges"]
    config.categorical_values = {
        "churn": ["Yes", "No"],
    }
    config.target_column = "churn"
    config.target_values = ["Yes", "No"]
    config.numerical_columns = [
        "tenure",
        "monthly_charges",
        "total_charges",
    ]
    config.id_column = "customer_id"

    config.report_path = "unused_report.json"
    config.status_file = "unused_status.json"

    return config


@pytest.fixture
def validator(config):
    return DataValidation(config)


@pytest.fixture
def valid_dataframe():
    return pd.DataFrame(
        {
            "customer_id": pd.Series(
                ["C001", "C002"],
                dtype="object",
            ),
            "tenure": pd.Series([12, 24], dtype="int64"),
            "monthly_charges": pd.Series(
                [50.5, 70.0],
                dtype="float64",
            ),
            "total_charges": pd.Series(
                [606.0, 1680.0],
                dtype="float64",
            ),
            "churn": pd.Series(
                ["No", "Yes"],
                dtype="object",
            ),
        }
    )


def test_dtype_matches_integer():
    assert DataValidation._dtype_matches(
        pd.Series([1, 2], dtype="int64"),
        "int",
    )


def test_dtype_matches_float():
    assert DataValidation._dtype_matches(
        pd.Series([1.5, 2.5]),
        "float",
    )


def test_integer_dtype_is_allowed_for_float():
    assert DataValidation._dtype_matches(
        pd.Series([1, 2], dtype="int64"),
        "float",
    )


def test_dtype_rejects_wrong_type():
    assert not DataValidation._dtype_matches(
        pd.Series(["one", "two"]),
        "int",
    )


def test_validate_columns_passes(validator, valid_dataframe):
    assert validator.validate_columns(valid_dataframe)


def test_validate_columns_detects_missing_column(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.drop(columns=["churn"])

    assert not validator.validate_columns(invalid)


def test_validate_columns_detects_unexpected_column(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.assign(unexpected_column=1)

    assert not validator.validate_columns(invalid)


def test_validate_dtypes_passes(validator, valid_dataframe):
    assert validator.validate_dtypes(valid_dataframe)


def test_validate_dtypes_detects_mismatch(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.copy()
    invalid["tenure"] = ["twelve", "twenty-four"]

    assert not validator.validate_dtypes(invalid)


def test_validate_missing_values_passes(
    validator,
    valid_dataframe,
):
    assert validator.validate_missing_values(valid_dataframe)


def test_validate_missing_values_respects_nullable_columns(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.copy()
    invalid.loc[0, "total_charges"] = float("nan")

    assert validator.validate_missing_values(invalid)


def test_validate_missing_values_rejects_non_nullable_missing(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.copy()
    invalid.loc[0, "customer_id"] = None

    assert not validator.validate_missing_values(invalid)


def test_validate_categorical_values_passes(
    validator,
    valid_dataframe,
):
    assert validator.validate_categorical_values(valid_dataframe)


def test_validate_categorical_values_rejects_invalid_label(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.copy()
    invalid.loc[0, "churn"] = "Maybe"

    assert not validator.validate_categorical_values(invalid)


def test_validate_numerical_ranges_passes(
    validator,
    valid_dataframe,
):
    assert validator.validate_numerical_ranges(valid_dataframe)


def test_validate_numerical_ranges_rejects_negative_values(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.copy()
    invalid.loc[0, "tenure"] = -1

    assert not validator.validate_numerical_ranges(invalid)


def test_validate_unique_ids_passes(
    validator,
    valid_dataframe,
):
    assert validator.validate_unique_ids(valid_dataframe)


def test_validate_unique_ids_rejects_duplicates(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.copy()
    invalid.loc[1, "customer_id"] = "C001"

    assert not validator.validate_unique_ids(invalid)


def test_validate_unique_ids_rejects_missing_id_column(
    validator,
    valid_dataframe,
):
    invalid = valid_dataframe.drop(columns=["customer_id"])

    assert not validator.validate_unique_ids(invalid)


def test_load_data_raises_on_read_failure(validator):
    with (
        patch(
            "customer_churn_prediction.components.data_validation.pd.read_csv",
            side_effect=OSError("file unavailable"),
        ),
        pytest.raises(DataValidationError),
    ):
        validator.load_data()
