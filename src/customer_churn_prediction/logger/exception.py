"""Project-specific exceptions for the customer churn prediction system."""


class CustomerChurnBaseError(Exception):
    """Base exception for the customer churn prediction project."""


class ChurnPredictionError(CustomerChurnBaseError):
    """Base exception for pipeline and application errors."""


class ConfigurationError(ChurnPredictionError):
    """Raised when configuration is invalid or missing."""


class DataIngestionError(ChurnPredictionError):
    """Raised when downloading or storing raw data fails."""


class DataValidationError(ChurnPredictionError):
    """Raised when data does not match the expected schema."""


class DataTransformationError(ChurnPredictionError):
    """Raised when cleaning, transforming, or splitting data fails."""


class ModelTrainingError(ChurnPredictionError):
    """Raised when model training or hyperparameter tuning fails."""


class ModelEvaluationError(ChurnPredictionError):
    """Raised when model evaluation fails."""


class ModelRegistryError(ChurnPredictionError):
    """Raised when registering or promoting a model fails."""


class ModelPackagingError(ChurnPredictionError):
    """Raised when packaging the model for serving fails."""


class ModelInferenceError(ChurnPredictionError):
    """Raised when model inference fails."""


class DatabaseError(ChurnPredictionError):
    """Raised when a database operation fails."""
