import pandas as pd
import matplotlib.pyplot as plt 
import re

# Load the dataset
df = pd.read_csv("data/spam.csv", encoding="latin1")

# Show the first 5 rows
print("----- FIRST 5 ROWS -----")
print(df.head())

# Show dataset size
print("\n----- DATASET SHAPE -----")
print(df.shape)

# Show column names
print("\n----- COLUMN NAMES -----")
print(df.columns)

# Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Check duplicate rows
print("\n----- DUPLICATE ROWS -----")
print(df.duplicated().sum())

# Check label distribution
print("\n----- LABEL DISTRIBUTION -----")
print(df["v1"].value_counts())

# Rename important columns
df = df.rename(columns={
    "v1":"Label",
    "v2":"Message"
})

# Remove unnecessary columns
df=df[["Label","Message"]]

print("\n----- CLEANED DATASET -----")
print(df.head())

print("\n----- CLEANED SHAPE -----")
print(df.shape)

print("\n----- CLEANED COLUMNS -----")
print(df.columns)

# Remove duplicate rows
df = df.drop_duplicates()

#df after removing duplicates
df.shape

print("\n----- AFTER REMOVING DUPLICATES -----")
print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_spam.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Message length analysis
df["message_length"] = df["Message"].str.len()

print("\n----- MESSAGE LENGTH -----")
print(df["message_length"].describe())

print("\n----- AVERAGE MESSAGE LENGTH BY LABEL -----")
print(df.groupby("Label")["message_length"].mean())

# Plot label distribution
df["Label"].value_counts().plot(kind="bar")

plt.title("Spam vs Ham Messages")
plt.xlabel("Label")
plt.ylabel("Number of Messages")
plt.show()

# Plot message length distribution
df["message_length"].plot(kind="hist", bins=50)

plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Number of Messages")
plt.show()

print("\n----- SAMPLE HAM MESSAGES -----")
print(df[df["Label"] == "ham"]["Message"].head(5).to_string(index=False))

print("\n----- SAMPLE SPAM MESSAGES -----")
print(df[df["Label"] == "spam"]["Message"].head(5).to_string(index=False))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("\n----- CLEANING TEST -----")

sample = "   WINNER!!    You Have WON a Prize!   "

print("Before:", sample)
print("After:", clean_text(sample))

# Apply text cleaning to the entire dataset
df["clean_message"] = df["Message"].apply(clean_text)

print("\n----- ORIGINAL VS CLEANED -----")
print(df[["Message", "clean_message"]].head(10).to_string(index=False))

# Check for empty messages after cleaning
empty_messages = (df["clean_message"] == "").sum()

print("\n----- EMPTY CLEANED MESSAGES -----")
print("Empty messages:", empty_messages)

# Show messages that became empty
print("\n----- EMPTY MESSAGE DETAILS -----")
print(df[df["clean_message"] == ""][["Label", "Message"]].to_string(index=False))

# Remove messages that became empty after cleaning
df = df[df["clean_message"] != ""]

print("\n----- AFTER REMOVING EMPTY MESSAGES -----")
print("Dataset shape:", df.shape)

# Save final cleaned dataset
df.to_csv("data/final_cleaned_spam.csv", index=False)

print("\n----- FINAL DATASET SAVED -----")
print("Final dataset shape:", df.shape)
print("Saved as: data/final_cleaned_spam.csv")