import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from customer_churn_prediction.entity.config_entity import ModelEvaluationConfig
from customer_churn_prediction.logger import ModelEvaluationError, get_logger
from customer_churn_prediction.utils.common import (
    classification_metrics,
    load_bin,
    load_json,
    save_json,
    save_status,
    setup_mlflow,
)


logger = get_logger("model_evaluation", "model_evaluation.log")

DEFAULT_THRESHOLD = 0.50


class ModelEvaluation:
    """Evaluate the trained model and track its results."""

    def __init__(self, config: ModelEvaluationConfig):
        """Initialize model evaluation with configuration."""

        self.config = config
        self.target_column = config.target_column

    def load_model(self):
        """Load the trained model from the artifacts directory."""

        try:
            file_path = self.config.model_path

            logger.info(f"Loading model: '{file_path}'")

            model = load_bin(file_path)

            logger.info("Trained model loaded successfully.")

            return model

        except Exception as e:
            logger.error(
                f"Model load failed - path='{self.config.model_path}': {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to load model") from e

    def load_test_data(self) -> pd.DataFrame:
        """Load the test dataset."""

        try:
            file_path = self.config.test_data_path

            logger.info(f"Reading test file: '{file_path}'")

            test = pd.read_csv(file_path)

            logger.info(f"Test data loaded successfully - shape='{test.shape}'")
            logger.info(f"Test columns: '{test.columns.tolist()}'")

            return test

        except Exception as e:
            logger.error(
                f"Test data load failed - path='{self.config.test_data_path}': {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to load test data") from e

    def prepare_test_data(
        self,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target from testing data."""

        try:
            if self.target_column not in test.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in testing data"
                )

            x_test = test.drop(columns=[self.target_column])
            y_test = test[self.target_column]

            logger.info(f"Test features shape: '{x_test.shape}'")
            logger.info(f"Test target shape: '{y_test.shape}'")

            return x_test, y_test

        except Exception as e:
            logger.error(
                f"Test data preparation failed: {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to prepare test data") from e

    def evaluate(
        self,
        model,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float,
    ) -> tuple[dict, pd.Series]:
        """Calculate classification metrics for the trained model."""

        try:
            proba = model.predict_proba(x_test)[:, 1]

            y_pred = (proba >= threshold).astype(int)

            metrics = classification_metrics(
                y_test,
                proba,
                threshold,
                self.config.cost_per_missed_churn,
                self.config.cost_per_unnecessary_offer,
            )

            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

            metrics["pr_auc"] = float(average_precision_score(y_test, proba))

            metrics["threshold"] = float(threshold)

            logger.info(f"Accuracy: '{metrics['accuracy']:.4f}'")
            logger.info(f"Precision: '{metrics['precision']:.4f}'")
            logger.info(f"Recall: '{metrics['recall']:.4f}'")
            logger.info(f"F1 Score: '{metrics['f1']:.4f}'")
            logger.info(f"ROC-AUC: '{metrics['roc_auc']:.4f}'")
            logger.info(f"PR-AUC: '{metrics['pr_auc']:.4f}'")

            save_json(
                self.config.metrics_path,
                metrics,
            )

            logger.info(f"Metrics saved: '{self.config.metrics_path}'")

            return metrics, y_pred

        except Exception as e:
            logger.error(
                f"Model evaluation failed: {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to evaluate model") from e

    def log_figure(self, fig, name: str):
        """Log evaluation figure to MLflow."""

        try:
            mlflow.log_figure(
                fig,
                f"evaluation_plots/{name}.png",
            )
            plt.close(fig)

        except Exception as e:
            logger.error(
                f"Figure logging failed - name='{name}': {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to log evaluation figure") from e

    def log_plots(
        self,
        y_test: pd.Series,
        proba,
        y_pred,
        threshold: float,
        sweep: pd.DataFrame,
    ):
        """Create and log model evaluation plots."""

        try:
            display = ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                display_labels=[
                    "No Churn",
                    "Churn",
                ],
            )

            display.ax_.set_title(f"Confusion Matrix (threshold {threshold:.2f})")

            self.log_figure(
                display.figure_,
                "confusion_matrix",
            )

            display = RocCurveDisplay.from_predictions(
                y_test,
                proba,
            )

            display.ax_.set_title("ROC Curve")

            self.log_figure(
                display.figure_,
                "roc_curve",
            )

            display = PrecisionRecallDisplay.from_predictions(
                y_test,
                proba,
            )

            display.ax_.set_title("Precision-Recall Curve")

            self.log_figure(
                display.figure_,
                "precision_recall_curve",
            )

            fig, ax = plt.subplots(figsize=(9, 6))

            for column in (
                "accuracy",
                "precision",
                "recall",
                "f1",
            ):
                ax.plot(
                    sweep["threshold"],
                    sweep[column],
                    marker="o",
                    label=column,
                )

            ax.axvline(
                threshold,
                linestyle="--",
                label=f"selected = {threshold:.2f}",
            )

            ax.set(
                title="Threshold Analysis",
                xlabel="Classification threshold",
                ylabel="Score",
            )

            ax.legend()

            self.log_figure(
                fig,
                "threshold_analysis",
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.plot(
                sweep["threshold"],
                sweep["estimated_total_cost"],
                marker="o",
            )

            ax.axvline(
                threshold,
                linestyle="--",
                label=f"selected = {threshold:.2f}",
            )

            ax.set(
                title="Estimated Total Cost by Threshold",
                xlabel="Classification threshold",
                ylabel="Estimated total cost",
            )

            ax.legend()

            self.log_figure(
                fig,
                "cost_analysis",
            )

        except Exception as e:
            logger.error(
                f"Evaluation plots failed: {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to create evaluation plots") from e

    def log_feature_importance(
        self,
        model,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """Calculate and log feature importance."""

        try:
            result = permutation_importance(
                model,
                x_test,
                y_test,
                scoring=self.config.importance_scoring,
                n_repeats=self.config.importance_repeats,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )

            importance = (
                pd.DataFrame(
                    {
                        "feature": x_test.columns,
                        "importance": result.importances_mean,
                    }
                )
                .sort_values(
                    "importance",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            fig, ax = plt.subplots(figsize=(8, 7))

            importance.head(15).sort_values("importance").plot(
                x="feature",
                y="importance",
                kind="barh",
                legend=False,
                ax=ax,
            )

            ax.set(
                title=(f"Permutation Importance ({self.config.importance_scoring})"),
                xlabel="Mean decrease in score",
                ylabel="Feature",
            )

            fig.tight_layout()

            self.log_figure(
                fig,
                "feature_importance",
            )

            mlflow.log_text(
                importance.to_csv(index=False),
                "evaluation_tables/feature_importance.csv",
            )

        except Exception as e:
            logger.error(
                f"Feature importance calculation failed: {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to calculate feature importance") from e

    def log_to_mlflow(
        self,
        model,
        metrics: dict,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred,
        proba,
        threshold: float,
        sweep: pd.DataFrame,
        summary: dict,
    ):
        """Log model evaluation results to the existing MLflow run."""

        try:
            with mlflow.start_run(run_id=summary["run_id"]):
                for name in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "estimated_total_cost",
                    "recall_gap",
                ):
                    if name in metrics:
                        mlflow.log_metric(
                            f"test_{name}",
                            metrics[name],
                        )

                mlflow.log_text(
                    classification_report(
                        y_test,
                        y_pred,
                        target_names=[
                            "No Churn",
                            "Churn",
                        ],
                        digits=4,
                        zero_division=0,
                    ),
                    "classification_report.txt",
                )

                self.log_plots(
                    y_test,
                    proba,
                    y_pred,
                    threshold,
                    sweep,
                )

                self.log_feature_importance(
                    model,
                    x_test,
                    y_test,
                )

                errors = x_test.copy()

                errors["y_true"] = y_test.values
                errors["y_pred"] = y_pred
                errors["y_proba"] = proba

                false_negatives = errors[
                    (errors["y_true"] == 1) & (errors["y_pred"] == 0)
                ]

                false_positives = errors[
                    (errors["y_true"] == 0) & (errors["y_pred"] == 1)
                ]

                mlflow.log_text(
                    false_negatives.to_csv(index=False),
                    "evaluation_tables/false_negatives.csv",
                )

                mlflow.log_text(
                    false_positives.to_csv(index=False),
                    "evaluation_tables/false_positives.csv",
                )

                run_id = mlflow.active_run().info.run_id

                logger.info(f"MLflow run_id: '{run_id}'")

        except Exception as e:
            logger.error(
                f"MLflow logging failed: {e}",
                exc_info=True,
            )
            raise ModelEvaluationError("Failed to log evaluation results") from e

    def run(self):
        """Run the model evaluation process."""

        try:
            setup_mlflow(self.config.mlflow)

            model = self.load_model()

            summary = load_json(self.config.summary_path)

            threshold = summary["threshold"]

            sweep = pd.read_csv(self.config.threshold_sweep_path)

            test = self.load_test_data()

            x_test, y_test = self.prepare_test_data(test)

            proba = model.predict_proba(x_test)[:, 1]

            metrics, y_pred = self.evaluate(
                model,
                x_test,
                y_test,
                threshold,
            )

            metrics["recall_gap"] = (
                summary["train_metrics"]["recall"] - metrics["recall"]
            )

            metrics["default_threshold_metrics"] = classification_metrics(
                y_test,
                proba,
                DEFAULT_THRESHOLD,
                self.config.cost_per_missed_churn,
                self.config.cost_per_unnecessary_offer,
            )

            save_json(self.config.metrics_path, metrics)
            save_status(self.config.status_file, "model_evaluation", True)

            self.log_to_mlflow(
                model, metrics, x_test, y_test, y_pred, proba, threshold, sweep, summary
            )

            logger.info(
                f"Test metrics at threshold "
                f"{threshold:.2f}: "
                f"recall='{metrics['recall']:.4f}', "
                f"precision='{metrics['precision']:.4f}', "
                f"ROC-AUC='{metrics['roc_auc']:.4f}', "
                f"PR-AUC='{metrics['pr_auc']:.4f}'"
            )

            logger.info("Model evaluation completed successfully")

            return metrics, y_pred, y_test

        except Exception as e:
            try:
                save_status(self.config.status_file, "model_evaluation", False, str(e))
            except Exception:
                logger.exception("Failed to save model evaluation status")
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ModelEvaluationError("Model evaluation process failed") from e
