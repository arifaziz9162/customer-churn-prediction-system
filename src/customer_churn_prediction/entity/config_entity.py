from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLflowConfig:
    repo_owner: str
    repo_name: str
    experiment_name: str


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_dataset: str
    kaggle_file_name: str
    download_dir: Path
    raw_data_path: Path
    status_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    raw_data_path: Path
    report_path: Path
    status_file: Path
    columns: dict
    id_column: str
    target_column: str
    target_values: list
    numerical_columns: list
    categorical_columns: list
    nullable_columns: list
    categorical_values: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    status_file: Path
    id_column: str
    target_column: str
    target_mapping: dict
    test_size: float
    random_state: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    model_path: Path
    summary_path: Path
    threshold_sweep_path: Path
    status_file: Path
    target_column: str
    numerical_columns: list
    categorical_columns: list
    categorical_values: dict
    random_state: int
    cv_folds: int
    scoring: str
    n_jobs: int
    n_trials: int
    seed: int
    direction: str
    search_space: dict
    min_threshold: float
    max_threshold: float
    threshold_step: float
    cost_per_missed_churn: float
    cost_per_unnecessary_offer: float
    mlflow: MLflowConfig


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    summary_path: Path
    threshold_sweep_path: Path
    metrics_path: Path
    status_file: Path
    target_column: str
    random_state: int
    n_jobs: int
    importance_scoring: str
    importance_repeats: int
    cost_per_missed_churn: float
    cost_per_unnecessary_offer: float
    mlflow: MLflowConfig


@dataclass(frozen=True)
class ModelRegistryConfig:
    root_dir: Path
    metrics_path: Path
    summary_path: Path
    report_path: Path
    model_name: str
    production_alias: str
    status_file: Path
    promotion_metric: str
    min_improvement: float
    mlflow: MLflowConfig


@dataclass(frozen=True)
class ModelPackagingConfig:
    root_dir: Path
    report_path: Path
    registry_model_name: str
    production_alias: str
    bento_model_name: str
    status_file: Path
    mlflow: MLflowConfig
