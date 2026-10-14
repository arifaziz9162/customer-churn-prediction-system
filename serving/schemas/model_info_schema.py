"""Response schema for deployed model information."""

from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Metadata returned by the model information endpoint."""

    status: str
    service: str
    service_version: str

    bento_model: str
    mlflow_version: str
    mlflow_run_id: str | None = None

    classification_threshold: float = Field(gt=0.0, lt=1.0)

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    roc_auc: float | None = None
