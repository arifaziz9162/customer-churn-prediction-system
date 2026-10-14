"""Human-readable labels for customer churn predictions."""

from serving.schemas import ChurnLabel


def get_churn_label(prediction: int) -> ChurnLabel:
    """Convert a binary prediction into a churn label."""
    if prediction not in (0, 1):
        raise ValueError("Prediction must be either 0 or 1.")

    return ChurnLabel.yes if prediction == 1 else ChurnLabel.no
