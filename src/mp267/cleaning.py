
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

def clean_data(df):
    df = df.copy()

    # Remove spaces and standardize vehicle names
    df["vehicle"] = (
        df["vehicle"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"\.+$", "", regex=True)
    )

    # Correct common abbreviated vehicle names
    df["vehicle"] = df["vehicle"].replace({
        "thr": "three_wheeler",
        "two": "two_wheeler",
    })

    # Correct Mall session duration unit mix-up
    mall = df["station_type"].eq("Mall")
    df.loc[mall, "session_minutes"] = (
        df.loc[mall, "session_minutes"] / 60
    )

    # Remove post-session information to avoid leakage
    df = df.drop(columns=["kwh_billed_in_app"], errors="ignore")

    # Correct target values recorded 10 times too large
    suspicious = df["energy_kwh"] > 100
    df.loc[suspicious, "energy_kwh"] = (
        df.loc[suspicious, "energy_kwh"] / 10
    )

    return df


if __name__ == "__main__":
    train_path = DATA_DIR / "train.csv"
    output_path = DATA_DIR / "train_clean.csv"

    train = pd.read_csv(train_path)
    cleaned = clean_data(train)

    cleaned.to_csv(output_path, index=False)

    print("Original train shape:", train.shape)
    print("Cleaned train shape:", cleaned.shape)
    print("Vehicle values:", sorted(cleaned["vehicle"].dropna().unique()))
    print("Saved to:", output_path)
