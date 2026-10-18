"""Generate sample student data, train a linear regression model, and save it."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"


def create_sample_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic study-hours and marks data for the practical."""
    random_generator = np.random.default_rng(seed=42)
    study_hours = random_generator.uniform(0, 10, size=100)
    noise = random_generator.normal(0, 5, size=100)
    marks = np.clip(5 + (9 * study_hours) + noise, 0, 100)
    return study_hours.reshape(-1, 1), marks


def train_and_save_model() -> None:
    """Train, evaluate, and persist the student marks prediction model."""
    study_hours, marks = create_sample_dataset()
    hours_train, hours_test, marks_train, marks_test = train_test_split(
        study_hours, marks, test_size=0.2, random_state=42
    )

    evaluation_model = LinearRegression()
    evaluation_model.fit(hours_train, marks_train)
    predictions = evaluation_model.predict(hours_test)

    print(f"Mean Absolute Error (MAE): {mean_absolute_error(marks_test, predictions):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(marks_test, predictions):.2f}")
    print(f"R-squared: {r2_score(marks_test, predictions):.2f}")

    final_model = LinearRegression()
    final_model.fit(study_hours, marks)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
