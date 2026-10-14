import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, pool

# Load environment variables from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Alembic configuration.
config = context.config

# Configure logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Validate required database environment variables.
required_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
missing_vars = [key for key in required_vars if not os.getenv(key)]

if missing_vars:
    raise RuntimeError(f"Missing database environment variables: {missing_vars}")

# Create the Customer Churn database URL.
db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    database=os.getenv("DB_NAME"),
)

# Migrations are managed manually; ORM metadata is not required.
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations without a database connection."""

    context.configure(
        url=db_url.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a database connection."""

    connectable = create_engine(db_url, poolclass=pool.NullPool)

    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)

            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


# Run migrations in the selected mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
