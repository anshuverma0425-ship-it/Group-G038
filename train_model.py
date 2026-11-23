
from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw" / "train.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "train_clean.csv"

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "MP267_EV_Energy_Prediction"
MODEL_NAME = "MP267_EV_Energy_Model"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    # Hash the ORIGINAL training dataset, not the cleaned file.
    data_hash = sha256_file(RAW_DATA)

    df = pd.read_csv(CLEAN_DATA)

    target = "energy_kwh"
    drop_columns = ["energy_kwh", "record_id", "date"]

    X = df.drop(columns=drop_columns, errors="ignore")
    y = df[target]

    # Convert categorical columns into numeric columns.
    X = pd.get_dummies(X, columns=X.select_dtypes(
        include=["object", "string", "category"]
    ).columns, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    configurations = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 300, "max_depth": None},
    ]

    results = []

    for index, params in enumerate(configurations, start=1):
        with mlflow.start_run(run_name=f"RandomForest_Config_{index}") as run:
            model = RandomForestRegressor(
                **params,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            mlflow.log_params(params)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
            results.append({
                "run_id": run.info.run_id,
                "mae": mae,
                "model": model,
                "params": params
            })

            print(f"Config {index}: MAE={mae:.4f}, R2={r2:.4f}")

    best = min(results, key=lambda item: item["mae"])

    print("\nBest configuration:", best["params"])
    print("Best MAE:", best["mae"])
    print("Best run ID:", best["run_id"])

    # Register the best run's logged model.
    model_uri = f"runs:/{best['run_id']}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    # Assign the required champion alias.
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        registered.version
    )

    # Save a local copy for later application use.
    import joblib
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best["model"], models_dir / "model.joblib")

    print("\nRegistered model:", MODEL_NAME)
    print("Champion version:", registered.version)
    print("Saved model to:", models_dir / "model.joblib")


if __name__ == "__main__":
    main()
