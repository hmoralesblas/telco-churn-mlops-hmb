import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


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

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5]
    }

    base_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    mlflow.set_experiment("fintech_churn_mlops")

    with mlflow.start_run(run_name="random_forest_gridsearch"):

        print("Ejecutando GridSearchCV (24 combinaciones × 5 folds)...")
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print(f"Mejores hiperparámetros: {best_params}")

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_params(best_params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)

        cv_results = grid_search.cv_results_
        for i in range(len(cv_results["params"])):
            with mlflow.start_run(run_name=f"combo_{i}", nested=True):
                mlflow.log_params(cv_results["params"][i])
                mlflow.log_metric("mean_cv_f1", cv_results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", cv_results["std_test_score"][i])
                mlflow.log_metric("rank", int(cv_results["rank_test_score"][i]))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "best_cv_f1": grid_search.best_score_,
            "best_params": best_params
        }

        with open(METRICS_PATH, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
            registered_model_name="TelcoChurnModel"
        )

        mlflow.log_artifact(str(METRICS_PATH))

        print("Modelo entrenado correctamente")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Best CV F1: {grid_search.best_score_:.4f}")
        print(f"Modelo guardado en: {MODEL_PATH}")
        print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    train_model()