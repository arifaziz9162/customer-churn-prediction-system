"""Integration tests for the customer churn ML pipeline."""

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import accuracy_score

from customer_churn_prediction.components import DataTransformation
from customer_churn_prediction.components import ModelTrainer


def make_customer_dataset(rows=40):
    """Create a synthetic customer churn dataset."""

    rng = np.random.default_rng(42)

    churn = np.array([0, 1] * (rows // 2))

    tenure = rng.integers(1, 73, size=rows)
    monthly_charges = rng.uniform(20, 120, size=rows)

    contracts = np.where(
        churn == 1,
        "Month-to-month",
        "Two year",
    )

    return pd.DataFrame(
        {
            "customerID": [f"C{i:04}" for i in range(rows)],
            "gender": rng.choice(["Male", "Female"], size=rows),
            "SeniorCitizen": rng.integers(0, 2, size=rows),
            "Partner": rng.choice(["Yes", "No"], size=rows),
            "Dependents": rng.choice(["Yes", "No"], size=rows),
            "tenure": tenure,
            "PhoneService": ["Yes"] * rows,
            "MultipleLines": rng.choice(
                ["Yes", "No"],
                size=rows,
            ),
            "InternetService": rng.choice(
                ["DSL", "Fiber optic", "No"],
                size=rows,
            ),
            "OnlineSecurity": rng.choice(
                ["Yes", "No", "No internet service"],
                size=rows,
            ),
            "OnlineBackup": rng.choice(
                ["Yes", "No", "No internet service"],
                size=rows,
            ),
            "DeviceProtection": rng.choice(
                ["Yes", "No", "No internet service"],
                size=rows,
            ),
            "TechSupport": rng.choice(
                ["Yes", "No", "No internet service"],
                size=rows,
            ),
            "StreamingTV": rng.choice(
                ["Yes", "No", "No internet service"],
                size=rows,
            ),
            "StreamingMovies": rng.choice(
                ["Yes", "No", "No internet service"],
                size=rows,
            ),
            "Contract": contracts,
            "PaperlessBilling": rng.choice(
                ["Yes", "No"],
                size=rows,
            ),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=rows,
            ),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": (tenure * monthly_charges).round(2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


@pytest.fixture
def raw_csv(tmp_path):
    """Create a synthetic raw dataset."""

    path = tmp_path / "customer_churn.csv"

    make_customer_dataset().to_csv(
        path,
        index=False,
    )

    return path


def test_transformation_and_training_integration(
    raw_csv,
    tmp_path,
):
    """Run real transformation and model fitting on temporary data."""

    transform_config = type("Config", (), {})()

    transform_config.raw_data_path = raw_csv
    transform_config.id_column = "customer_id"
    transform_config.target_column = "churn"
    transform_config.target_mapping = {
        "No": 0,
        "Yes": 1,
    }
    transform_config.test_size = 0.25
    transform_config.random_state = 42
    transform_config.train_data_path = tmp_path / "train.csv"
    transform_config.test_data_path = tmp_path / "test.csv"
    transform_config.status_file = tmp_path / "status.json"

    transformer = DataTransformation(transform_config)

    loaded = transformer.load_data()
    cleaned = transformer.clean_data(loaded)
    train, test = transformer.split_data(cleaned)
    transformer.save_data(train, test)

    assert "customer_id" not in cleaned.columns
    assert set(cleaned["churn"].unique()) == {0, 1}
    assert len(train) + len(test) == len(cleaned)

    trainer_config = type("Config", (), {})()

    trainer_config.target_column = "churn"
    trainer_config.random_state = 42

    trainer_config.numerical_columns = [
        "senior_citizen",
        "tenure",
        "monthly_charges",
        "total_charges",
    ]

    trainer_config.categorical_columns = [
        column
        for column in train.columns
        if column
        not in {
            "senior_citizen",
            "tenure",
            "monthly_charges",
            "total_charges",
            "churn",
        }
    ]

    trainer_config.categorical_values = {
        column: sorted(train[column].dropna().unique().tolist())
        for column in trainer_config.categorical_columns
    }

    model_trainer = ModelTrainer(trainer_config)

    train_loaded = model_trainer.load_data() if False else train.copy()

    x_train, y_train = model_trainer.prepare_train_data(train_loaded)

    pipeline = model_trainer.build_pipeline(
        {
            "n_estimators": 10,
            "max_depth": 5,
        }
    )

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_train)

    assert len(predictions) == len(y_train)
    assert set(predictions).issubset({0, 1})

    accuracy = accuracy_score(y_train, predictions)

    assert 0.0 <= accuracy <= 1.0

    assert transform_config.train_data_path.is_file()
    assert transform_config.test_data_path.is_file()
