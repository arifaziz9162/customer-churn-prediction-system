"""Unit tests for the customer churn model trainer."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from customer_churn_prediction.components import (
    ModelTrainer,
)
from customer_churn_prediction.logger import ModelTrainingError


@pytest.fixture
def config():
    config = type("Config", (), {})()

    config.target_column = "churn"
    config.random_state = 42

    config.numerical_columns = [
        "tenure",
        "monthly_charges",
    ]

    config.categorical_columns = ["contract"]

    config.categorical_values = {
        "contract": [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    }

    config.cv_folds = 2
    config.n_trials = 1
    config.scoring = "roc_auc"
    config.n_jobs = 1
    config.direction = "maximize"
    config.seed = 42

    config.search_space = {
        "N_ESTIMATORS": [10, 20],
        "MAX_DEPTH": [2, 5],
        "MIN_SAMPLES_SPLIT": [2, 4],
        "MIN_SAMPLES_LEAF": [1, 2],
        "MAX_FEATURES": ["sqrt"],
        "CLASS_WEIGHT": [None],
    }

    config.min_threshold = 0.2
    config.max_threshold = 0.8
    config.threshold_step = 0.1
    config.cost_per_missed_churn = 100.0
    config.cost_per_unnecessary_offer = 10.0

    config.model_path = None
    config.threshold_sweep_path = None
    config.train_data_path = None

    return config


@pytest.fixture
def trainer(config):
    return ModelTrainer(config)


@pytest.fixture
def training_data():
    x = pd.DataFrame(
        {
            "tenure": [
                1,
                2,
                3,
                4,
                10,
                12,
                20,
                24,
                30,
                36,
                48,
                60,
            ],
            "monthly_charges": [
                90,
                85,
                95,
                88,
                80,
                75,
                70,
                65,
                60,
                55,
                45,
                40,
            ],
            "contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "One year",
                "Two year",
                "One year",
                "Two year",
                "Two year",
                "One year",
                "Two year",
                "Two year",
            ],
        }
    )

    y = pd.Series(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        name="churn",
    )

    return x, y


def test_prepare_train_data_separates_features_and_target(
    trainer,
    training_data,
):
    x, y = training_data

    frame = x.copy()
    frame["churn"] = y

    features, target = trainer.prepare_train_data(frame)

    assert "churn" not in features.columns
    assert target.name == "churn"
    assert len(features) == len(target)


def test_prepare_train_data_rejects_missing_target(
    trainer,
    training_data,
):
    x, _ = training_data

    with pytest.raises(ModelTrainingError):
        trainer.prepare_train_data(x)


def test_build_preprocessor_returns_column_transformer(
    trainer,
):
    from sklearn.compose import ColumnTransformer

    preprocessor = trainer.build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_build_pipeline_returns_pipeline(trainer):
    pipeline = trainer.build_pipeline({"n_estimators": 10, "max_depth": 3})

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_can_fit_and_predict(
    trainer,
    training_data,
):
    x, y = training_data

    pipeline = trainer.build_pipeline({"n_estimators": 10, "max_depth": 3})

    pipeline.fit(x, y)

    predictions = pipeline.predict(x)
    probabilities = pipeline.predict_proba(x)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(y), 2)
    assert np.isfinite(probabilities).all()


def test_load_data_wraps_file_error(trainer):
    """Test that a CSV read error is wrapped in ModelTrainingError."""
    with (
        patch(
            "customer_churn_prediction.components.model_trainer.pd.read_csv",
            side_effect=OSError("missing file"),
        ),
        pytest.raises(ModelTrainingError),
    ):
        trainer.load_data()
