
import pandas as pd

file_path = "data/Phishing_Email.csv"

print("Loading email dataset...")

df = pd.read_csv(
    file_path,
    encoding="latin1",
    low_memory=False
)

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nEmail type counts:")
print(df["Email Type"].value_counts(dropna=False))

print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum())
