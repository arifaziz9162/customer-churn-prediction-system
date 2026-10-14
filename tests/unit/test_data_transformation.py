"""Unit tests for customer churn data transformation."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from customer_churn_prediction.components.data_transformation import (
    DataTransformation,
)
from customer_churn_prediction.logger import DataTransformationError


@pytest.fixture
def config(tmp_path):
    config = MagicMock()

    config.raw_data_path = tmp_path / "raw.csv"
    config.id_column = "customer_id"
    config.target_column = "churn"
    config.target_mapping = {"No": 0, "Yes": 1}
    config.test_size = 0.25
    config.random_state = 42

    config.train_data_path = tmp_path / "train.csv"
    config.test_data_path = tmp_path / "test.csv"
    config.status_file = tmp_path / "status.json"

    return config


@pytest.fixture
def transformer(config):
    return DataTransformation(config)


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03}" for i in range(20)],
            "tenure": list(range(1, 21)),
            "monthly_charges": [50.0 + i for i in range(20)],
            "churn": ["No", "Yes"] * 10,
        }
    )


def test_clean_data_removes_customer_id(
    transformer,
    dataframe,
):
    result = transformer.clean_data(dataframe)

    assert "customer_id" not in result.columns


def test_clean_data_encodes_target(
    transformer,
    dataframe,
):
    result = transformer.clean_data(dataframe)

    assert set(result["churn"].unique()) == {0, 1}
    assert pd.api.types.is_integer_dtype(result["churn"])


def test_clean_data_does_not_modify_original(
    transformer,
    dataframe,
):
    original = dataframe.copy()

    transformer.clean_data(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_clean_data_rejects_unknown_target(
    transformer,
    dataframe,
):
    invalid = dataframe.copy()
    invalid.loc[0, "churn"] = "Unknown"

    with pytest.raises(DataTransformationError):
        transformer.clean_data(invalid)


def test_split_data_preserves_all_rows(
    transformer,
    dataframe,
):
    cleaned = transformer.clean_data(dataframe)

    train, test = transformer.split_data(cleaned)

    assert len(train) + len(test) == len(cleaned)


def test_split_data_preserves_target_distribution(
    transformer,
    dataframe,
):
    cleaned = transformer.clean_data(dataframe)

    train, test = transformer.split_data(cleaned)

    assert set(train["churn"].unique()) == {0, 1}
    assert set(test["churn"].unique()) == {0, 1}


def test_split_data_is_reproducible(
    transformer,
    dataframe,
):
    cleaned = transformer.clean_data(dataframe)

    train_a, test_a = transformer.split_data(cleaned)
    train_b, test_b = transformer.split_data(cleaned)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_split_data_wraps_invalid_input(
    transformer,
):
    invalid = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(DataTransformationError):
        transformer.split_data(invalid)


def test_save_data_writes_train_and_test(
    transformer,
    dataframe,
):
    train = dataframe.iloc[:15]
    test = dataframe.iloc[15:]

    transformer.save_data(train, test)

    assert transformer.config.train_data_path.is_file()
    assert transformer.config.test_data_path.is_file()

    loaded_train = pd.read_csv(transformer.config.train_data_path)
    loaded_test = pd.read_csv(transformer.config.test_data_path)

    assert len(loaded_train) == len(train)
    assert len(loaded_test) == len(test)


def test_load_data_wraps_missing_file(
    transformer,
):
    with pytest.raises(DataTransformationError):
        transformer.load_data()
