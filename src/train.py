import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


PROCESSED_DATA_PATH = Path("data/processed_churn.csv")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")


def load_processed_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset procesado: {path}")
    return pd.read_csv(path)


def train_model() -> None:
    print("Iniciando entrenamiento del modelo...")

    df = load_processed_data(PROCESSED_DATA_PATH)

    target_column = "churn"

    if target_column not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    mlflow.set_experiment("fintech_churn_mlops")

    with mlflow.start_run(run_name="random_forest_churn_model"):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        with open(METRICS_PATH, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        mlflow.log_artifact(str(METRICS_PATH))

        print("Modelo entrenado correctamente")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Modelo guardado en: {MODEL_PATH}")
        print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()