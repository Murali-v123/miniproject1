
import pandas as pd

df = pd.read_csv("data/cleaned_email.csv")

print("Total email examples:", len(df))

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nReview examples labeled as phishing:")
phishing = df[df["label"] == "phishing"].sample(
    n=min(10, len(df[df["label"] == "phishing"])),
    random_state=42
)

for i, (_, row) in enumerate(phishing.iterrows(), start=1):
    print(f"\nExample {i}:")
    print(str(row["message"])[:300])

print("\nReview examples labeled as legitimate:")
legitimate = df[df["label"] == "legitimate"].sample(
    n=min(5, len(df[df["label"] == "legitimate"])),
    random_state=42
)

for i, (_, row) in enumerate(legitimate.iterrows(), start=1):
    print(f"\nExample {i}:")
    print(str(row["message"])[:300])
