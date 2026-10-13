
import pandas as pd

df = pd.read_csv("data/combined_dataset.csv")

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nEmpty messages:")
print((df["message"].astype(str).str.strip() == "").sum())

print("\nDuplicate messages:")
print(df["message"].duplicated().sum())

print("\nSample messages:")
for _, row in df.groupby("label").head(1).iterrows():
    print(f"\nLabel: {row['label']}")
    print(f"Message: {str(row['message'])[:200]}")
