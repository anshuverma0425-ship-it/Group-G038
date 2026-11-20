
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"

train = pd.read_csv(DATA / "train.csv")
later = pd.read_csv(DATA / "later.csv")
holdout = pd.read_csv(DATA / "holdout_features.csv")

print("DATASET OVERVIEW")
print("Train:", train.shape)
print("Later:", later.shape)
print("Holdout:", holdout.shape)

print("\n1. LEAKING COLUMN")
leaks = [
    c for c in train.columns
    if c not in holdout.columns and c != "energy_kwh"
]
print(leaks)

print("\n2. UNIT MIXUP")
print(train.groupby("station_type")["session_minutes"].agg(
    ["count", "median", "mean", "min", "max"]
))
mall = train[train["station_type"] == "Mall"]["session_minutes"]
print("Mall values divisible by 60:", (mall % 60 == 0).all())

print("\n3. SENTINEL VALUES")
sentinels = [-9999, -999, -99, -1, 999, 9999]
found = {}
for col in train.select_dtypes(include="number").columns:
    for value in sentinels:
        count = int((train[col] == value).sum())
        if count:
            found[(col, value)] = count
print(found if found else "No sentinel values found")

print("\n4. DUPLICATE ROWS")
duplicates = int(train.duplicated().sum())
print(duplicates)

print("\n5. STUCK SENSOR")
train["date"] = pd.to_datetime(train["date"])
numeric_cols = train.select_dtypes(include="number").columns
stuck = []

groups = [(None, train)] + list(train.groupby("station_type"))

for group_name, subset in groups:
    for col in numeric_cols:
        daily = subset.groupby("date")[col].agg(
            lambda x: x.iloc[0] if x.nunique() == 1 else None
        )
        start = None
        previous_date = None
        previous_value = None

        for date, value in daily.items():
            if pd.isna(value):
                start = None
                previous_date = None
                previous_value = None
                continue

            consecutive = (
                previous_date is not None
                and (date - previous_date).days == 1
                and value == previous_value
            )

            if not consecutive:
                start = date

            if (date - start).days >= 2:
                stuck.append({
                    "column": col,
                    "group": group_name,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": date.strftime("%Y-%m-%d"),
                    "value": value
                })

            previous_date = date
            previous_value = value

print(stuck if stuck else "No stuck sensor found")

print("\n6. TARGET TYPOS")
typos = train[train["energy_kwh"] > 100]
print("Suspicious rows:", len(typos))
print(typos[["record_id", "energy_kwh"]].to_string(index=False))

print("\n7. SPELLING VARIANTS")
print("Raw vehicle values:")
print(train["vehicle"].value_counts())
print("\nClean vehicle values from later.csv:")
print(sorted(later["vehicle"].unique()))
print("True levels:", later["vehicle"].nunique())
