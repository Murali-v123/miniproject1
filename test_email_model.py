
import joblib

# Load the saved email model and its matching vectorizer
model = joblib.load("model/email_model.pkl")
vectorizer = joblib.load("model/email_tfidf_vectorizer.pkl")

print("Email phishing detector is ready!")
print("Type an email to classify it.")
print("Type 'exit' to stop.\n")

while True:
    message = input("Enter email text: ").strip()

    if message.lower() == "exit":
        print("See you, bro!")
        break

    if not message:
        print("Please enter some text.\n")
        continue

    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]

    print("Model prediction:", prediction)

    if prediction == "phishing":
        print("Warning: This resembles examples labeled phishing.")
    else:
        print("This resembles examples labeled legitimate.")

    print()
