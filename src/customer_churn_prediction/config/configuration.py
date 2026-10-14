from pathlib import Path

from dotenv import load_dotenv

from customer_churn_prediction.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from customer_churn_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    MLflowConfig,
    ModelEvaluationConfig,
    ModelPackagingConfig,
    ModelRegistryConfig,
    ModelTrainerConfig,
)
from customer_churn_prediction.logger import ConfigurationError
from customer_churn_prediction.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH,
        schema_path: Path = SCHEMA_FILE_PATH,
    ):
        load_dotenv()
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.schema = read_yaml(schema_path)

    def get_mlflow_config(self) -> MLflowConfig:
        try:
            cfg = self.config["MLFLOW"]
            return MLflowConfig(
                repo_owner=cfg["REPO_OWNER"],
                repo_name=cfg["REPO_NAME"],
                experiment_name=cfg["EXPERIMENT_NAME"],
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key {e} in MLFLOW config") from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Build configuration for the data ingestion stage."""

        try:
            cfg = self.config["DATA_INGESTION"]
            root_dir = Path(cfg["ROOT_DIR"])
            download_dir = Path(cfg["DOWNLOAD_DIR"])
            create_directories([root_dir, download_dir])
            return DataIngestionConfig(
                root_dir=root_dir,
                kaggle_dataset=cfg["KAGGLE_DATASET"],
                kaggle_file_name=cfg["KAGGLE_FILE_NAME"],
                download_dir=download_dir,
                raw_data_path=Path(cfg["RAW_DATA_PATH"]),
                status_file=Path(cfg["STATUS_FILE"]),
            )

        except KeyError as e:
            raise ConfigurationError(
                f"Missing key {e} in DATA_INGESTION configuration"
            ) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            cfg = self.config["DATA_VALIDATION"]
            create_directories([Path(cfg["ROOT_DIR"])])
            return DataValidationConfig(
                root_dir=Path(cfg["ROOT_DIR"]),
                raw_data_path=Path(cfg["RAW_DATA_PATH"]),
                report_path=Path(cfg["REPORT_PATH"]),
                status_file=Path(cfg["STATUS_FILE"]),
                columns=self.schema["COLUMNS"],
                id_column=self.schema["ID_COLUMN"],
                target_column=self.schema["TARGET_COLUMN"],
                target_values=list(self.schema["TARGET_MAPPING"].keys()),
                numerical_columns=self.schema["NUMERICAL_COLUMNS"],
                categorical_columns=self.schema["CATEGORICAL_COLUMNS"],
                nullable_columns=self.schema["NULLABLE_COLUMNS"],
                categorical_values=self.schema["CATEGORICAL_VALUES"],
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key {e} in validation config") from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            cfg = self.config["DATA_TRANSFORMATION"]
            training = self.params["TRAINING"]
            create_directories([Path(cfg["ROOT_DIR"])])
            return DataTransformationConfig(
                root_dir=Path(cfg["ROOT_DIR"]),
                raw_data_path=Path(cfg["RAW_DATA_PATH"]),
                train_data_path=Path(cfg["TRAIN_DATA_PATH"]),
                test_data_path=Path(cfg["TEST_DATA_PATH"]),
                status_file=Path(cfg["STATUS_FILE"]),
                id_column=self.schema["ID_COLUMN"],
                target_column=self.schema["TARGET_COLUMN"],
                target_mapping=self.schema["TARGET_MAPPING"],
                test_size=training["TEST_SIZE"],
                random_state=training["RANDOM_STATE"],
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key {e} in transformation config") from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            cfg = self.config["MODEL_TRAINER"]
            training = self.params["TRAINING"]
            bayes = self.params["BAYESIAN_OPTIMIZATION"]
            threshold = self.params["THRESHOLD_SELECTION"]
            create_directories([Path(cfg["ROOT_DIR"])])
            return ModelTrainerConfig(
                root_dir=Path(cfg["ROOT_DIR"]),
                train_data_path=Path(cfg["TRAIN_DATA_PATH"]),
                model_path=Path(cfg["MODEL_PATH"]),
                summary_path=Path(cfg["SUMMARY_PATH"]),
                threshold_sweep_path=Path(cfg["THRESHOLD_SWEEP_PATH"]),
                status_file=Path(cfg["STATUS_FILE"]),
                target_column=self.schema["TARGET_COLUMN"],
                numerical_columns=self.schema["NUMERICAL_COLUMNS"],
                categorical_columns=self.schema["CATEGORICAL_COLUMNS"],
                categorical_values=self.schema["CATEGORICAL_VALUES"],
                random_state=training["RANDOM_STATE"],
                cv_folds=training["CV_FOLDS"],
                scoring=training["SCORING"],
                n_jobs=training["N_JOBS"],
                n_trials=bayes["N_TRIALS"],
                seed=bayes["SEED"],
                direction=bayes["DIRECTION"],
                search_space=self.params["RANDOM_FOREST_SEARCH_SPACE"],
                min_threshold=threshold["MIN_THRESHOLD"],
                max_threshold=threshold["MAX_THRESHOLD"],
                threshold_step=threshold["STEP"],
                cost_per_missed_churn=threshold["COST_PER_MISSED_CHURN"],
                cost_per_unnecessary_offer=threshold["COST_PER_UNNECESSARY_OFFER"],
                mlflow=self.get_mlflow_config(),
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key {e} in trainer config") from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            cfg = self.config["MODEL_EVALUATION"]
            training = self.params["TRAINING"]
            evaluation = self.params["MODEL_EVALUATION"]
            threshold = self.params["THRESHOLD_SELECTION"]
            create_directories([Path(cfg["ROOT_DIR"])])
            return ModelEvaluationConfig(
                root_dir=Path(cfg["ROOT_DIR"]),
                test_data_path=Path(cfg["TEST_DATA_PATH"]),
                model_path=Path(cfg["MODEL_PATH"]),
                summary_path=Path(cfg["SUMMARY_PATH"]),
                threshold_sweep_path=Path(cfg["THRESHOLD_SWEEP_PATH"]),
                metrics_path=Path(cfg["METRICS_PATH"]),
                status_file=Path(cfg["STATUS_FILE"]),
                target_column=self.schema["TARGET_COLUMN"],
                random_state=training["RANDOM_STATE"],
                n_jobs=training["N_JOBS"],
                importance_scoring=evaluation["IMPORTANCE_SCORING"],
                importance_repeats=evaluation["IMPORTANCE_REPEATS"],
                cost_per_missed_churn=threshold["COST_PER_MISSED_CHURN"],
                cost_per_unnecessary_offer=threshold["COST_PER_UNNECESSARY_OFFER"],
                mlflow=self.get_mlflow_config(),
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key {e} in evaluation config") from e

    def get_model_registry_config(self) -> ModelRegistryConfig:
        try:
            cfg = self.config["MODEL_REGISTRY"]
            registry = self.params["MODEL_REGISTRY"]
            create_directories([Path(cfg["ROOT_DIR"])])
            return ModelRegistryConfig(
                root_dir=Path(cfg["ROOT_DIR"]),
                metrics_path=Path(cfg["METRICS_PATH"]),
                summary_path=Path(cfg["SUMMARY_PATH"]),
                report_path=Path(cfg["REPORT_PATH"]),
                model_name=cfg["MODEL_NAME"],
                production_alias=cfg["PRODUCTION_ALIAS"],
                status_file=Path(cfg["STATUS_FILE"]),
                promotion_metric=registry["PROMOTION_METRIC"],
                min_improvement=registry["MIN_IMPROVEMENT"],
                mlflow=self.get_mlflow_config(),
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key {e} in registry config") from e

    def get_model_packaging_config(self) -> ModelPackagingConfig:
        try:
            cfg = self.config["MODEL_PACKAGING"]
            create_directories([Path(cfg["ROOT_DIR"])])
            return ModelPackagingConfig(
                root_dir=Path(cfg["ROOT_DIR"]),
                report_path=Path(cfg["REPORT_PATH"]),
                registry_model_name=cfg["REGISTRY_MODEL_NAME"],
                production_alias=cfg["PRODUCTION_ALIAS"],
                bento_model_name=cfg["BENTO_MODEL_NAME"],
                mlflow=self.get_mlflow_config(),
                status_file=Path(cfg["STATUS_FILE"]),
            )
        except KeyError as e:
            raise ConfigurationError(f"Missing key {e} in packaging config") from e
