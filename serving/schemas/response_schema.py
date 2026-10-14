"""Response schemas for customer churn predictions."""

from enum import Enum

from pydantic import BaseModel, Field


class ChurnLabel(str, Enum):
    """Human-readable churn prediction labels."""

    yes = "Yes"
    no = "No"


class PredictionResponse(BaseModel):
    """Response returned after a churn prediction."""

    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: int = Field(ge=0, le=1)
    churn_label: ChurnLabel
    classification_threshold: float = Field(gt=0.0, lt=1.0)
    message: str
    model_version: str
