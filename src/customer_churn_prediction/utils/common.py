import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from box import ConfigBox
from ensure import ensure_annotations
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from customer_churn_prediction.entity.config_entity import MLflowConfig
from customer_churn_prediction.logger import (
    ConfigurationError,
    CustomerChurnBaseError,
    get_logger,
)

logger = get_logger("common", "common.log")


@ensure_annotations
def read_yaml(path: str | Path) -> ConfigBox:
    """Read a YAML file and return as ConfigBox."""

    try:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"YAML file does not exist: '{path}'")

        with open(path, encoding="utf-8") as file:
            content = yaml.safe_load(file)

        if content is None:
            raise ValueError(f"YAML file is empty: '{path}'")

        if not isinstance(content, dict):
            raise ValueError(f"YAML file must contain a mapping: '{path}'")

        logger.info(f"YAML file loaded successfully: '{path}'")

        return ConfigBox(content)

    except Exception as e:
        logger.error(f"Failed to read YAML file '{path}': {e}", exc_info=True)
        raise ConfigurationError(f"Failed to read YAML: '{path}'") from e


def create_directories(
    paths: list[str | Path] | None = None,
    verbose: bool = True,
):
    """Create directories if they don't exist."""

    try:
        if not paths:
            return

        for path in paths:
            os.makedirs(path, exist_ok=True)

            if verbose:
                logger.info(f"Created directory at: '{path}'")

    except Exception as e:
        logger.error(f"Failed to create directory '{path}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to create directories: '{path}'") from e


@ensure_annotations
def save_json(path: str | Path, data: dict):
    """Save dictionary as JSON file."""

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, default=_json_default)

        logger.info(f"JSON file saved at: '{path}'")

    except Exception as e:
        logger.error(f"Failed to save JSON file '{path}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to save JSON: '{path}'") from e


@ensure_annotations
def load_json(path: str | Path) -> ConfigBox:
    """Load JSON file as ConfigBox."""

    try:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"JSON file does not exist: '{path}'")

        with open(path, encoding="utf-8") as file:
            content = json.load(file)

        logger.info(f"JSON file loaded successfully: '{path}'")

        return ConfigBox(content)

    except Exception as e:
        logger.error(f"Failed to load JSON file '{path}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to load JSON: '{path}'") from e


def save_bin(data: Any, path: str | Path):
    """Save binary object using joblib."""

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: '{path}'")

    except Exception as e:
        logger.error(f"Failed to save binary file '{path}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to save binary: '{path}'") from e


def load_bin(path: str | Path) -> Any:
    """Load binary object using joblib."""

    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Binary file does not exist: '{path}'")
        data = joblib.load(path)
        logger.info(f"Binary file loaded successfully: '{path}'")
        return data

    except Exception as e:
        logger.error(f"Failed to load binary file '{path}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to load binary: '{path}'") from e


def save_status(
    path: str | Path,
    stage: str,
    success: bool,
    error: str | None = None,
) -> None:
    """Save the execution status of a pipeline stage."""

    try:
        path = Path(path)

        status_data = {
            "stage": stage,
            "status": "success" if success else "failed",
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }

        if not success:
            status_data["error"] = error or "An unexpected error occurred"

        save_json(path, status_data)

        logger.info(
            f"Status file saved at: '{path}' "
            f"for stage: '{stage}' "
            f"with status: '{status_data['status']}'"
        )

    except Exception as e:
        logger.error(f"Failed to save status file '{path}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to save status: '{path}'") from e


@ensure_annotations
def get_size(path: str | Path) -> str:
    """Get file size in KB."""

    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist at: '{path}'")
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~{size_in_kb} KB"

    except Exception as e:
        logger.error(f"Failed to get file size '{path}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to get size: '{path}'") from e


def _json_default(value: Any) -> Any:
    """Convert NumPy and Path objects for JSON."""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def to_snake_case(name: str) -> str:
    """Convert column name to snake_case."""

    try:
        return re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])",
            "_",
            name,
        ).lower()

    except Exception as e:
        logger.error(f"Failed to convert name to snake_case '{name}': {e}", exc_info=True)
        raise CustomerChurnBaseError(f"Failed to convert name: '{name}'") from e


def classification_metrics(
    y_true,
    y_proba,
    threshold: float,
    cost_per_missed_churn: float,
    cost_per_unnecessary_offer: float,
) -> dict:
    """Calculate classification metrics at a given threshold."""

    try:
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "estimated_total_cost": float(
                fn * cost_per_missed_churn + fp * cost_per_unnecessary_offer
            ),
        }

    except Exception as e:
        logger.error(f"Failed to calculate classification metrics: {e}", exc_info=True)
        raise CustomerChurnBaseError(
            "Failed to calculate classification metrics"
        ) from e


def setup_mlflow(mlflow_config: MLflowConfig):
    """Configure MLflow tracking and experiment."""

    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            import dagshub

            dagshub.init(
                repo_owner=mlflow_config.repo_owner,
                repo_name=mlflow_config.repo_name,
                mlflow=True,
            )

        mlflow.set_experiment(mlflow_config.experiment_name)

        logger.info(
            f"MLflow configured successfully: '{mlflow_config.experiment_name}'"
        )

    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}", exc_info=True)
        raise CustomerChurnBaseError("Failed to setup MLflow") from e
