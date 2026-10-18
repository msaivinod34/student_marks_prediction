"""Flask web application for predicting student marks."""

import math
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"
MIN_STUDY_HOURS = 0
MAX_STUDY_HOURS = 10

model = None


def load_model():
    """Load the model produced by train_model.py."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run 'python train_model.py' before starting the Flask app."
        )
    return joblib.load(MODEL_PATH)


def get_model():
    """Return the loaded model, creating it lazily when needed."""
    global model
    if model is None:
        model = load_model()
    return model


app = Flask(__name__)


@app.get("/")
def home():
    """Show the study-hours prediction form."""
    return render_template("index.html")


@app.get("/health")
def health():
    """Report whether the web application and its model are ready."""
    try:
        model_loaded = get_model() is not None
    except FileNotFoundError:
        model_loaded = False
    status = "ok" if model_loaded else "error"
    return jsonify(status=status, model_loaded=model_loaded)


@app.post("/prediction")
def prediction():
    """Validate submitted study hours and return the predicted marks."""
    try:
        prediction_model = get_model()
    except FileNotFoundError as exc:
        return _model_unavailable_error(str(exc))

    if request.is_json:
        values = request.get_json(silent=True)
        study_hours = values.get("study_hours") if isinstance(values, dict) else None
    else:
        study_hours = request.form.get("study_hours")

    if study_hours is None or not str(study_hours).strip():
        return _validation_error("Please enter your study hours.")
    if isinstance(study_hours, bool):
        return _validation_error("Study hours must be a number.")

    try:
        hours = float(study_hours)
    except (TypeError, ValueError):
        return _validation_error("Study hours must be a number.")

    if not math.isfinite(hours):
        return _validation_error("Study hours must be a finite number.")
    if not MIN_STUDY_HOURS <= hours <= MAX_STUDY_HOURS:
        return _validation_error(
            f"Study hours must be between {MIN_STUDY_HOURS} and "
            f"{MAX_STUDY_HOURS} hours."
        )

    predicted_marks = round(float(prediction_model.predict([[hours]])[0]), 2)
    if request.is_json:
        return jsonify(study_hours=hours, predicted_marks=predicted_marks)

    return render_template(
        "index.html",
        study_hours=hours,
        predicted_marks=predicted_marks,
    )


def _model_unavailable_error(message: str):
    if request.is_json:
        return jsonify(error=message), 503
    return (
        render_template(
            "index.html",
            error=message,
            study_hours=request.form.get("study_hours", ""),
        ),
        503,
    )


def _validation_error(message: str):
    if request.is_json:
        return jsonify(error=message), 400
    return (
        render_template(
            "index.html",
            error=message,
            study_hours=request.form.get("study_hours", ""),
        ),
        400,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
