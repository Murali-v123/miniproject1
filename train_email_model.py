
import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix

# Load the prepared email dataset
df = pd.read_csv("data/cleaned_email.csv")
df = df.dropna(subset=["message", "label"])
df["message"] = df["message"].astype(str).str.strip()
df = df[df["message"] != ""]

# Keep only the two email categories
df = df[df["label"].isin(["legitimate", "phishing"])]

# Split data before fitting the vectorizer
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the email classifier
model = ComplementNB()
model.fit(X_train_tfidf, y_train)

# Evaluate on the held-out test split
predictions = model.predict(X_test_tfidf)

print("\nEmail model evaluation")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix (label order: legitimate, phishing):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["legitimate", "phishing"]
))

# Save the new email model separately
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/email_model.pkl")
joblib.dump(vectorizer, "model/email_tfidf_vectorizer.pkl")

print("\nSaved:")
print("model/email_model.pkl")
print("model/email_tfidf_vectorizer.pkl")
