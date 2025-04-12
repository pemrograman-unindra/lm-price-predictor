import json
import pandas as pd

# Load JSON file
with open("data/lm_price.json", "r") as f:
    raw_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(raw_data, columns=["dateTime", "price"])
df["dateTime"] = pd.to_datetime(df["dateTime"], unit="ms")
df = df[["dateTime", "price"]]
df.rename(columns={"price": "price"}, inplace=True)

# Save to CSV format compatible with the project structure
csv_path = "data/lm_price.csv"
df.to_csv(csv_path, index=False)

csv_path
