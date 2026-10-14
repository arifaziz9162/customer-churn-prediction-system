"""Request and response schemas for prediction history."""

from typing import Any

from pydantic import BaseModel, Field

from serving.config import settings


class PredictionsRequest(BaseModel):
    """Parameters for retrieving stored prediction history."""

    limit: int = Field(
        default=20,
        ge=1,
        le=settings.PREDICTIONS_MAX_LIMIT,
    )


class PredictionsResponse(BaseModel):
    """Stored prediction history response."""

    count: int = Field(ge=0)
    predictions: list[dict[str, Any]]
