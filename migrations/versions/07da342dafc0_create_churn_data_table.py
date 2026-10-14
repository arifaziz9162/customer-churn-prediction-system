"""create churn data table

Revision ID: 07da342dafc0
Revises:
Create Date: 2026-10-10 19:59:04.611787

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "07da342dafc0"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create churn_data table."""

    op.create_table(
        "churn_data",
        sa.Column("customer_id", sa.String(32), primary_key=True),
        sa.Column("gender", sa.String(16), nullable=False),
        sa.Column("senior_citizen", sa.Integer(), nullable=False),
        sa.Column("partner", sa.String(8), nullable=False),
        sa.Column("dependents", sa.String(8), nullable=False),
        sa.Column("tenure", sa.Integer(), nullable=False),
        sa.Column("phone_service", sa.String(8), nullable=False),
        sa.Column("multiple_lines", sa.String(32), nullable=False),
        sa.Column("internet_service", sa.String(32), nullable=False),
        sa.Column("online_security", sa.String(32), nullable=False),
        sa.Column("online_backup", sa.String(32), nullable=False),
        sa.Column("device_protection", sa.String(32), nullable=False),
        sa.Column("tech_support", sa.String(32), nullable=False),
        sa.Column("streaming_tv", sa.String(32), nullable=False),
        sa.Column("streaming_movies", sa.String(32), nullable=False),
        sa.Column("contract", sa.String(32), nullable=False),
        sa.Column("paperless_billing", sa.String(8), nullable=False),
        sa.Column("payment_method", sa.String(64), nullable=False),
        sa.Column("monthly_charges", sa.Float(), nullable=False),
        sa.Column("total_charges", sa.Float(), nullable=True),
        sa.Column("churn", sa.String(8), nullable=False),
    )


def downgrade() -> None:
    """Drop churn_data table."""

    op.drop_table("churn_data")
