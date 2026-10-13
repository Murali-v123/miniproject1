
import pandas as pd

# Load the cleaned SMS dataset
sms_df = pd.read_csv("data/final_cleaned_spam.csv")

# Keep the message text and label
sms_df = sms_df[["Message", "Label"]].copy()
sms_df.columns = ["message", "label"]

# Convert SMS labels to our project's format
sms_df["label"] = sms_df["label"].map({
    "ham": "legitimate",
    "spam": "spam"
})

# Load the cleaned email dataset
email_df = pd.read_csv("data/cleaned_email.csv")

# Keep the message text and label
email_df = email_df[["message", "label"]].copy()

# Keep only the three categories we need
combined_df = pd.concat(
    [sms_df, email_df],
    ignore_index=True
)

# Remove missing, empty, and duplicate messages
combined_df = combined_df.dropna(subset=["message", "label"])
combined_df["message"] = combined_df["message"].astype(str).str.strip()
combined_df = combined_df[combined_df["message"] != ""]
combined_df = combined_df.drop_duplicates(subset=["message"])

# Save to a NEW file without changing the original datasets
combined_df.to_csv("data/combined_dataset.csv", index=False)

print("Combined dataset created successfully!")
print("\nDataset shape:", combined_df.shape)
print("\nLabel counts:")
print(combined_df["label"].value_counts())
print("\nSaved to data/combined_dataset.csv")
