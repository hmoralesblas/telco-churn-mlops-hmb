from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = Path("models/model.pkl")
PROCESSED_DATA_PATH = Path("data/processed_churn.csv")

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_expected_columns():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df.drop(columns=["churn"]).columns.tolist()


model = load_model()
expected_columns = get_expected_columns()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API MLOps Churn activa",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Debe enviar un JSON válido"}), 400

    input_df = pd.DataFrame([data])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(
        columns=expected_columns,
        fill_value=0
    )

    prediction = model.predict(input_encoded)[0]

    result = "Churn" if int(prediction) == 1 else "No Churn"

    return jsonify({
        "prediction": int(prediction),
        "prediction_label": result
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)