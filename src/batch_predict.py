from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/model.pkl")
PROCESSED_DATA_PATH = Path("data/processed_churn.csv")
OUTPUT_PATH = Path("data/batch_predictions.csv")


def main() -> None:
    print("Iniciando inferencia batch...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset procesado: {PROCESSED_DATA_PATH}")

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=["churn"])

    predictions = model.predict(X)

    df_output = df.copy()
    df_output["prediction"] = predictions
    df_output["prediction_label"] = df_output["prediction"].map({
        0: "No Churn",
        1: "Churn"
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_PATH, index=False)

    print(f"Inferencia batch completada.")
    print(f"Registros procesados: {len(df_output)}")
    print(f"Archivo generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()