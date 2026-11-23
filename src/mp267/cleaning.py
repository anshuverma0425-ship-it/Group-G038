
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def clean_data(df):
    df = df.copy()

    # 1. Standardize vehicle names
    df["vehicle"] = (
        df["vehicle"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"\.+$", "", regex=True)
    )

    # 2. Correct abbreviated vehicle names
    df["vehicle"] = df["vehicle"].replace({
        "thr": "three_wheeler",
        "two": "two_wheeler",
    })

    # 3. Correct Mall session duration (seconds to minutes)
    mall = df["station_type"].eq("Mall")
    df.loc[mall, "session_minutes"] /= 60

    # 4. Remove leaking feature
    df = df.drop(
        columns=["kwh_billed_in_app"],
        errors="ignore"
    )

    # 5. Correct target values recorded 10 times too large
    suspicious = df["energy_kwh"] > 100
    df.loc[suspicious, "energy_kwh"] /= 10

    return df


if __name__ == "__main__":

    train_path = RAW_DIR / "train.csv"
    output_path = PROCESSED_DIR / "train_clean.csv"

    train = pd.read_csv(train_path)

    cleaned = clean_data(train)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(output_path, index=False)

    print("CLEANING COMPLETE")
    print("Original shape:", train.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Original data preserved:", train_path.exists())
    print("Vehicle categories:",
          sorted(cleaned["vehicle"].dropna().unique()))
    print("Mall median:",
          cleaned.loc[cleaned["station_type"] == "Mall",
                      "session_minutes"].median())
    print("Remaining suspicious targets:",
          int((cleaned["energy_kwh"] > 100).sum()))
    print("Saved to:", output_path)
