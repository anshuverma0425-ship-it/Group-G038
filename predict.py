
import pandas as pd
import joblib

# Load the already-trained model
model = joblib.load("models/energy_model.joblib")

print("\n=== EV Charging Energy Predictor ===")
print("Enter your EV charging details:\n")

battery = float(input("Battery start percentage (0-100): "))
duration = float(input("Session duration in minutes: "))
charger = float(input("Charger power in kW: "))
temperature = float(input("Temperature in Celsius: "))
age = float(input("Vehicle age in years: "))
queue = int(input("Queue before charging: "))
vehicle = input("Vehicle type (car/two_wheeler/three_wheeler): ").strip()
station = input("Station type (Residential/Highway/Mall): ").strip()

# Prepare input in the same format used during training
user_data = pd.DataFrame([{
    "battery_start_pct": battery,
    "session_minutes": duration,
    "charger_kw": charger,
    "temp_c": temperature,
    "vehicle_age_yr": age,
    "queue_before": queue,
    "vehicle": vehicle,
    "station_type": station
}])

# Generate prediction
prediction = model.predict(user_data)[0]

print("\n=== Prediction Result ===")
print(f"Predicted energy consumption: {prediction:.2f} kWh")
