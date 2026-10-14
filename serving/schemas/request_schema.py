"""Request schemas for customer churn predictions."""

from pydantic import BaseModel, ConfigDict, Field


class CustomerChurnInput(BaseModel):
    """Validated customer features for churn prediction."""

    model_config = ConfigDict(extra="forbid")

    gender: str = Field(description="Customer gender")
    senior_citizen: int = Field(ge=0, le=1)
    partner: str
    dependents: str
    tenure: int = Field(ge=0)
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
