
import pandas as pd

# Load the two prepared datasets
sms_df = pd.read_csv("data/final_cleaned_spam.csv")
email_df = pd.read_csv("data/cleaned_email.csv")

# Standardize the SMS columns and labels
sms_df = sms_df[["Message", "Label"]].copy()
sms_df.columns = ["message", "label"]

sms_df["label"] = sms_df["label"].map({
    "ham": "legitimate",
    "spam": "spam"
})

# Add source information
sms_df["source"] = "SMS"
email_df = email_df[["message", "label"]].copy()
email_df["source"] = "Email"

# Combine for inspection only
df = pd.concat([sms_df, email_df], ignore_index=True)

print("Label counts by source:\n")
print(pd.crosstab(df["label"], df["source"]))

print("\nTotal rows by source:")
print(df["source"].value_counts())

print("\nNote: This check does not modify any files.")

