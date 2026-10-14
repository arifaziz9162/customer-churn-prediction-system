"""create prediction history table

Revision ID: f1fd6cb99ede
Revises: 07da342dafc0
Create Date: 2026-10-12 15:13:26.020902

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1fd6cb99ede"
down_revision: str | Sequence[str] | None = "07da342dafc0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the table used to store prediction history."""

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
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
        sa.Column("total_charges", sa.Float(), nullable=False),
        sa.Column("predicted_churn", sa.Boolean(), nullable=False),
        sa.Column("churn_probability", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])


def downgrade() -> None:
    """Remove the prediction history table."""

    op.drop_index(
        "ix_predictions_created_at",
        table_name="predictions",
    )
    op.drop_table("predictions")
