
import pandas as pd

# Load the original email dataset
df = pd.read_csv(
    "data/Phishing_Email.csv",
    encoding="latin1",
    low_memory=False
)

# Keep only the text and its label
df = df[["Email Text", "Email Type"]].copy()

# Rename columns for easier use
df.columns = ["message", "label"]

# Remove rows with missing or empty messages
df = df.dropna(subset=["message", "label"])
df["message"] = df["message"].astype(str).str.strip()
df = df[df["message"] != ""]

# Convert labels to our project's format
df["label"] = df["label"].map({
    "Safe Email": "legitimate",
    "Phishing Email": "phishing"
})

# Remove any rows with unrecognized labels
df = df.dropna(subset=["label"])

# Remove duplicate email examples
df = df.drop_duplicates(subset=["message"])

# Save to a NEW file
df.to_csv("data/cleaned_email.csv", index=False)

print("Email dataset prepared successfully!")
print("\nDataset shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nSaved to data/cleaned_email.csv")
