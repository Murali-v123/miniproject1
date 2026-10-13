
import pandas as pd

df = pd.read_csv("data/combined_dataset.csv")

for label in ["legitimate", "spam", "phishing"]:
    print("\n" + "=" * 60)
    print("CATEGORY:", label.upper())
    print("=" * 60)

    samples = df[df["label"] == label].sample(
        n=min(3, len(df[df["label"] == label])),
        random_state=42
    )

    for i, (_, row) in enumerate(samples.iterrows(), start=1):
        print(f"\nExample {i}:")
        print(str(row["message"])[:400])
