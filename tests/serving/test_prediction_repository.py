"""Tests for prediction repository persistence."""

from concurrent.futures import Future
from unittest.mock import Mock, patch

import numpy as np
import pytest

from serving.persistence.prediction_repository import PredictionRepository


@pytest.fixture
def features():
    """Return sample customer features containing NumPy scalars."""

    return [
        "Female",
        np.int64(0),
        "Yes",
        "No",
        np.int64(12),
        "Yes",
        "No",
        "DSL",
        "Yes",
        "No",
        "No",
        "No",
        "No",
        "No",
        "Month-to-month",
        "Yes",
        "Electronic check",
        np.float64(70.0),
        np.float64(850.0),
    ]


def test_numpy_scalar_conversion():
    """Convert NumPy scalar values to native Python values."""

    assert PredictionRepository._to_python_value(np.int64(12)) == 12
    assert PredictionRepository._to_python_value(np.float64(70.0)) == 70.0
    assert PredictionRepository._to_python_value("Female") == "Female"


def test_save_writes_native_python_values(features):
    """Persist normalized values through the database operations layer."""

    repository = PredictionRepository.__new__(PredictionRepository)
    mock_database = Mock()

    with patch(
        "serving.persistence.prediction_repository.DatabaseOperations",
        return_value=mock_database,
    ):
        repository.save(features, np.int64(1), np.float64(0.5853))

    saved_features, saved_prediction, saved_probability = (
        mock_database.save_prediction.call_args.args
    )

    assert saved_features[1] == 0
    assert type(saved_features[1]) is int

    assert saved_features[-2] == 70.0
    assert type(saved_features[-2]) is float

    assert saved_prediction == 1
    assert type(saved_prediction) is int

    assert saved_probability == pytest.approx(0.5853)
    assert type(saved_probability) is float

    mock_database.save_prediction.assert_called_once()


def test_save_propagates_database_failure(features):
    """Propagate database failures to the caller."""

    repository = PredictionRepository.__new__(PredictionRepository)
    mock_database = Mock()
    mock_database.save_prediction.side_effect = RuntimeError("Database failed")

    with patch(
        "serving.persistence.prediction_repository.DatabaseOperations",
        return_value=mock_database,
    ):
        with pytest.raises(RuntimeError, match="Database failed"):
            repository.save(features, 1, 0.5853)

    mock_database.save_prediction.assert_called_once()


def test_save_async_submits_background_write(features):
    """Submit a persistence operation to the executor."""

    repository = PredictionRepository.__new__(PredictionRepository)
    repository._executor = Mock()

    future = Mock(spec=Future)
    repository._executor.submit.return_value = future

    with patch.object(repository, "_on_saved") as on_saved:
        result = repository.save_async(features, 1, 0.5853)

    repository._executor.submit.assert_called_once_with(
        repository.save,
        features,
        1,
        0.5853,
    )

    assert result is future
    future.add_done_callback.assert_called_once_with(on_saved)
