import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/raw/thyroid.csv")

print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")