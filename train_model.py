
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned data
data = pd.read_csv("data/train_clean.csv")

target = "energy_kwh"

numeric_features = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
]

categorical_features = [
    "vehicle",
    "station_type",
]

features = numeric_features + categorical_features

X = data[features]
y = data[target]

# Split data: 80% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
    ],
    remainder="passthrough",
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )),
])

# Train
model.fit(X_train, y_train)

# Evaluate on validation data
predictions = model.predict(X_val)

metrics = {
    "validation_rows": int(len(y_val)),
    "MAE": float(mean_absolute_error(y_val, predictions)),
    "RMSE": float(mean_squared_error(y_val, predictions) ** 0.5),
    "R2": float(r2_score(y_val, predictions)),
}

# Save model and metrics
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/energy_model.joblib")

with open("models/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMODEL TRAINING COMPLETE")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print("\nValidation metrics:")

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nModel saved: models/energy_model.joblib")
print("Metrics saved: models/metrics.json")
