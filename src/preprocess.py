import pandas as pd
from pathlib import Path

try:
    PROJECT_ROOT = Path("telco_customer_churn_mlops.csv").resolve().parent.parent
    
except NameError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

#RAW_DATA_PATH = Path("data/telco_customer_churn_mlops.csv")
#PROCESSED_DATA_PATH = Path("data/processed_churn.csv")
#RAW_DATA_PATH = PROJECT_ROOT/"data/telco_customer_churn_mlops.csv"
#PROCESSED_DATA_PATH = PROJECT_ROOT/"data/processed_churn.csv"

RAW_DATA_PATH = PROJECT_ROOT/"data/telco_customer_churn_mlops.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT/"data/processed_churn.csv"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RAW_DATA_PATH:", RAW_DATA_PATH)
print("EXISTS:", RAW_DATA_PATH.exists())

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
        print(f"Archivo no encontrado en {path}, creando un DataFrame de ejemplo para pruebas.")
        
        df_sample = pd.DataFrame({"customer_id": [1, 2], "monthly_charges": [70.5, 80.0], "tenure_months": [12, 24], "support_tickets_last_6m": [1, 0], "senior_citizen": [0, 1], "churn": ["Yes", "No"]})
        return df_sample

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())

    df["tenure_months"] = pd.to_numeric(df["tenure_months"], errors="coerce")
    df["tenure_months"] = df["tenure_months"].fillna(df["tenure_months"].median())

    df["support_tickets_last_6m"] = pd.to_numeric(df["support_tickets_last_6m"], errors="coerce")
    df["support_tickets_last_6m"] = df["support_tickets_last_6m"].fillna(0)

    df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce")
    df["senior_citizen"] = df["senior_citizen"].fillna(0)

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    return df_encoded


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    print("Iniciando preprocesamiento...")

    df = load_data(RAW_DATA_PATH)
    print(f"Dataset original: {df.shape}")
    print(f"Columnas originales: {list(df.columns)}")

    df_clean = clean_data(df)
    df_processed = encode_features(df_clean)

    save_processed_data(df_processed, PROCESSED_DATA_PATH)

    print(f"Dataset procesado: {df_processed.shape}")
    print(f"Archivo generado: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()