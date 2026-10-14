"""Tests for customer churn label conversion."""

import pytest

from serving.schemas import ChurnLabel
from serving.services.quality_label import get_churn_label


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (0, ChurnLabel.no),
        (1, ChurnLabel.yes),
    ],
)
def test_get_churn_label_returns_expected_label(prediction, expected):
    """Return the correct label for a binary churn prediction."""
    assert get_churn_label(prediction) == expected


@pytest.mark.parametrize("prediction", [2, -1, 3, 100])
def test_get_churn_label_rejects_invalid_prediction(prediction):
    """Reject predictions outside the binary classification range."""
    with pytest.raises(
        ValueError,
        match="Prediction must be either 0 or 1",
    ):
        get_churn_label(prediction)


def test_get_churn_label_rejects_none():
    """Reject a missing prediction."""
    with pytest.raises(ValueError):
        get_churn_label(None)


def test_get_churn_label_returns_enum():
    """Return a ChurnLabel enum instead of a plain string."""
    result = get_churn_label(1)

    assert isinstance(result, ChurnLabel)
