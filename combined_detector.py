
import joblib

# Load the saved SMS model and vectorizer
sms_model = joblib.load("model/spam_model.pkl")
sms_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Load the saved email model and vectorizer
email_model = joblib.load("model/email_model.pkl")
email_vectorizer = joblib.load("model/email_tfidf_vectorizer.pkl")

print("=" * 40)
print("     COMBINED MESSAGE DETECTOR")
print("=" * 40)
print("1. SMS message")
print("2. Email message")
print("Type 'exit' to quit.")

while True:
    choice = input("\nChoose SMS or Email (1/2): ").strip()

    if choice.lower() == "exit":
        print("Goodbye, bro!")
        break

    if choice not in ["1", "2"]:
        print("Please enter 1 for SMS or 2 for Email.")
        continue

    message = input("Enter your message: ").strip()

    if message.lower() == "exit":
        print("Goodbye, bro!")
        break

    if not message:
        print("Please enter some text.")
        continue

    if choice == "1":
        features = sms_vectorizer.transform([message])
        prediction = sms_model.predict(features)[0]

        if prediction == "spam":
            result = "spam"
        else:
            result = "legitimate"

        print("\nPrediction:", result)

    else:
        features = email_vectorizer.transform([message])
        prediction = email_model.predict(features)[0]

        print("\nPrediction:", prediction)

        if prediction == "phishing":
            print("Warning: This resembles examples labeled phishing.")
        else:
            print("This resembles examples labeled legitimate.")

    print("Reminder: Predictions can be wrong. Check suspicious messages carefully.")
