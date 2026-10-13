import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from customer_churn_prediction.entity.config_entity import ModelTrainerConfig
from customer_churn_prediction.logger import ModelTrainingError, get_logger
from customer_churn_prediction.utils.common import (
    classification_metrics,
    save_bin,
    save_json,
    save_status,
    setup_mlflow,
)


logger = get_logger("model_trainer", "model_trainer.log")


class ModelTrainer:
    """Find the best Random Forest parameters and train the final model."""

    def __init__(self, config: ModelTrainerConfig):
        """Initialize model trainer with configuration."""

        self.config = config
        self.target_column = config.target_column

    def load_data(self) -> pd.DataFrame:
        """Load training data."""

        try:
            file_path = self.config.train_data_path
            logger.info(f"Reading train_file: '{file_path}'")
            train = pd.read_csv(file_path)
            logger.info(f"Training data loaded successfully - shape='{train.shape}'")
            logger.info(f"Training columns: '{train.columns.tolist()}'")
            return train

        except Exception as e:
            logger.error(
                f"Training data loaded failed - "
                f"path='{self.config.train_data_path}': {e}",
                exc_info=True,
            )
            raise ModelTrainingError("Failed to load training data") from e

    def prepare_train_data(
        self,
        train: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target from training data."""

        try:
            if self.target_column not in train.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in training data"
                )

            x_train = train.drop(columns=[self.target_column])
            y_train = train[self.target_column]

            logger.info(f"Features shape: '{x_train.shape}'")
            logger.info(f"Target shape: '{y_train.shape}'")

            return x_train, y_train

        except Exception as e:
            logger.error(f"Training data preparation failed: {e}", exc_info=True)
            raise ModelTrainingError("Failed to prepare training data") from e

    def build_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing steps for numerical and categorical columns."""

        try:
            categories = [
                sorted(self.config.categorical_values[column])
                for column in self.config.categorical_columns
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical",
                        SimpleImputer(strategy="median"),
                        self.config.numerical_columns,
                    ),
                    (
                        "categorical",
                        OneHotEncoder(
                            categories=categories,
                            drop="first",
                            handle_unknown="ignore",
                        ),
                        self.config.categorical_columns,
                    ),
                ],
                remainder="drop",
                sparse_threshold=0.0,
                verbose_feature_names_out=False,
            )

            return preprocessor

        except Exception as e:
            logger.error(f"Preprocessor creation failed: {e}", exc_info=True)
            raise ModelTrainingError("Failed to create data preprocessor") from e

    def build_pipeline(self, params: dict) -> Pipeline:
        """Create preprocessing and Random Forest model pipeline."""

        try:
            model = RandomForestClassifier(
                random_state=self.config.random_state, n_jobs=1, **params
            )

            pipeline = Pipeline(
                steps=[("preprocessor", self.build_preprocessor()), ("model", model)]
            )

            return pipeline

        except Exception as e:
            logger.error(f"Model pipeline creation failed: {e}", exc_info=True)
            raise ModelTrainingError("Failed to create model pipeline") from e

    def optimize(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[dict, float]:
        """Find the best Random Forest parameters using Optuna."""

        try:
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

            logger.info(
                f"Starting Optuna optimization - "
                f"trials='{self.config.n_trials}' "
                f"cv='{self.config.cv_folds}' "
                f"scoring='{self.config.scoring}'"
            )

            def objective(trial: optuna.Trial) -> float:
                space = self.config.search_space

                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", *space["N_ESTIMATORS"]
                    ),
                    "max_depth": trial.suggest_int("max_depth", *space["MAX_DEPTH"]),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", *space["MIN_SAMPLES_SPLIT"]
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", *space["MIN_SAMPLES_LEAF"]
                    ),
                    "max_features": trial.suggest_categorical(
                        "max_features", space["MAX_FEATURES"]
                    ),
                    "class_weight": trial.suggest_categorical(
                        "class_weight", space["CLASS_WEIGHT"]
                    ),
                }

                pipeline = self.build_pipeline(params)

                scores = cross_val_score(
                    pipeline,
                    x_train,
                    y_train,
                    cv=cv,
                    scoring=self.config.scoring,
                    n_jobs=self.config.n_jobs,
                )

                return float(scores.mean())

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(
                direction=self.config.direction,
                sampler=optuna.samplers.TPESampler(seed=self.config.seed),
            )

            study.optimize(objective, n_trials=self.config.n_trials)

            logger.info(f"Optimization completed - best_score='{study.best_value:.4f}'")
            logger.info(f"Best parameters: '{study.best_params}'")

            return (study.best_params, float(study.best_value))

        except Exception as e:
            logger.error(
                f"Hyperparameter optimization failed: {e}",
                exc_info=True,
            )
            raise ModelTrainingError("Failed to optimize model parameters") from e

    def select_threshold(
        self,
        params: dict,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[pd.DataFrame, float]:
        """Select the classification threshold using out-of-fold predictions."""

        try:
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            pipeline = self.build_pipeline(params)
            oof_proba = cross_val_predict(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=self.config.n_jobs,
            )[:, 1]

            thresholds = np.round(
                np.arange(
                    self.config.min_threshold,
                    self.config.max_threshold + self.config.threshold_step / 2,
                    self.config.threshold_step,
                ),
                4,
            )

            rows = []

            for threshold in thresholds:
                metrics = classification_metrics(
                    y_train,
                    oof_proba,
                    threshold,
                    self.config.cost_per_missed_churn,
                    self.config.cost_per_unnecessary_offer,
                )

                rows.append({"threshold": float(threshold), **metrics})

            sweep = pd.DataFrame(rows)
            best = sweep.loc[sweep["estimated_total_cost"].idxmin()]
            threshold = float(best["threshold"])
            logger.info(
                f"Selected threshold: '{threshold:.2f}' "
                f"estimated_cost='{best['estimated_total_cost']:.0f}'"
            )

            return sweep, threshold

        except Exception as e:
            logger.error(
                f"Threshold selection failed: {e}",
                exc_info=True,
            )
            raise ModelTrainingError("Failed to select classification threshold") from e

    def train_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        best_params: dict,
        threshold: float,
        sweep: pd.DataFrame,
    ) -> dict:
        """Train the final Random Forest model and log it to MLflow."""

        try:
            logger.info(
                f"Training final Random Forest model with parameters: '{best_params}'"
            )

            model = self.build_pipeline(best_params)
            model.fit(x_train, y_train)
            train_metrics = classification_metrics(
                y_train,
                model.predict_proba(x_train)[:, 1],
                threshold,
                self.config.cost_per_missed_churn,
                self.config.cost_per_unnecessary_offer,
            )

            self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
            save_bin(model, self.config.model_path)

            self.config.threshold_sweep_path.parent.mkdir(parents=True, exist_ok=True)
            sweep.to_csv(self.config.threshold_sweep_path, index=False)
            logger.info(f"Final model saved: '{self.config.model_path}'")
            return {"model": model, "train_metrics": train_metrics}

        except Exception as e:
            logger.error(f"Final model training failed: {e}", exc_info=True)
            raise ModelTrainingError("Failed to train final model") from e

    def run(self) -> dict:
        """Run the model training process."""

        try:
            setup_mlflow(self.config.mlflow)
            train = self.load_data()
            x_train, y_train = self.prepare_train_data(train)
            best_params, best_cv_score = self.optimize(x_train, y_train)
            sweep, threshold = self.select_threshold(best_params, x_train, y_train)
            oof_metrics = sweep.loc[sweep["threshold"] == threshold].iloc[0].to_dict()

            result = self.train_model(x_train, y_train, best_params, threshold, sweep)

            model = result["model"]
            train_metrics = result["train_metrics"]

            with mlflow.start_run(run_name="random_forest_optuna") as run:
                mlflow.log_params(
                    {
                        "model": "RandomForest",
                        "optimization_method": "Optuna TPE",
                        "n_trials": self.config.n_trials,
                        "cv_strategy": "StratifiedKFold",
                        "cv_folds": self.config.cv_folds,
                        "scoring": self.config.scoring,
                        "classification_threshold": threshold,
                        "threshold_selection_method": ("cost_minimizing_oof"),
                        "cost_per_missed_churn": (self.config.cost_per_missed_churn),
                        "cost_per_unnecessary_offer": (
                            self.config.cost_per_unnecessary_offer
                        ),
                    }
                )

                mlflow.log_params(best_params)

                mlflow.log_metric(f"cv_{self.config.scoring}", best_cv_score)

                mlflow.log_metric(
                    "oof_estimated_total_cost", oof_metrics["estimated_total_cost"]
                )

                for metric in ("accuracy", "precision", "recall", "f1"):
                    mlflow.log_metric(f"train_{metric}", train_metrics[metric])

                mlflow.log_artifact(
                    str(self.config.threshold_sweep_path),
                    artifact_path="evaluation_tables",
                )

                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    serialization_format=(
                        mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                    ),
                    signature=infer_signature(
                        x_train,
                        model.predict(x_train),
                    ),
                    input_example=x_train.head(5),
                )
                run_id = run.info.run_id
                model_uri = model_info.model_uri

                summary = {
                    "run_id": run_id,
                    "model_uri": model_uri,
                    "scoring": self.config.scoring,
                    "best_cv_score": best_cv_score,
                    "best_params": best_params,
                    "threshold": threshold,
                    "train_metrics": train_metrics,
                }

                save_json(self.config.summary_path, summary)

                save_status(self.config.status_file, "model_trainer", True)

            logger.info(
                f"Model training completed successfully - MLflow run: '{run_id}'"
            )

            return summary

        except Exception as e:
            try:
                save_status(self.config.status_file, "model_trainer", False, str(e))
            except Exception:
                logger.exception("Failed to save model trainer status")
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ModelTrainingError("Model training process failed") from e
