import os
from urllib.parse import quote_plus

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

from customer_churn_prediction.logger import DatabaseError, get_logger

load_dotenv()

logger = get_logger("db_connection", "database.log")


def get_connection():
    """Create and return a PostgreSQL connection."""

    try:
        required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
        missing = [key for key in required if not os.getenv(key)]

        if missing:
            raise ValueError(f"Missing database environment variables: {missing}")

        conn = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
        )
        logger.info("PostgreSQL connection established successfully.")
        return conn

    except Exception as e:
        logger.error("PostgreSQL connection failed: %s", e, exc_info=True)
        raise DatabaseError("Failed to connect to PostgreSQL.") from e


def get_engine():
    """Create and return a SQLAlchemy engine."""

    try:
        required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
        missing = [key for key in required if not os.getenv(key)]

        if missing:
            raise ValueError(f"Missing database environment variables: {missing}")

        user = quote_plus(os.getenv("DB_USER"))
        password = quote_plus(os.getenv("DB_PASSWORD"))
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        database = quote_plus(os.getenv("DB_NAME"))

        database_url = (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )
        engine = create_engine(database_url)

        logger.info("SQLAlchemy engine created successfully.")
        return engine

    except Exception as e:
        logger.error("SQLAlchemy engine creation failed: %s", e, exc_info=True)
        raise DatabaseError("Failed to create SQLAlchemy engine.") from e
