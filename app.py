
import streamlit as st
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Page setup
st.set_page_config(
    page_title="Message Safety Detector",
    page_icon="🛡️",
    layout="centered"
)

# Load saved models
@st.cache_resource
def load_models():
    sms_model = joblib.load(MODEL_DIR / "spam_model.pkl")
    sms_vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    email_model = joblib.load(MODEL_DIR / "email_model.pkl")
    email_vectorizer = joblib.load(
        MODEL_DIR / "email_tfidf_vectorizer.pkl"
    )
    return sms_model, sms_vectorizer, email_model, email_vectorizer


st.title("🛡️ Message Safety Detector")
st.write(
    "Check an SMS or email message for potential spam or phishing."
)

st.info(
    "This is a machine-learning prototype. Its predictions can be wrong."
)

try:
    sms_model, sms_vectorizer, email_model, email_vectorizer = load_models()

    message_type = st.selectbox(
        "Choose message type",
        ["SMS Message", "Email Message"]
    )

    message = st.text_area(
        "Paste your message here",
        height=180,
        placeholder="Type or paste your SMS or email..."
    )

    if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Bro, please enter a message first.")
        else:
            if message_type == "SMS Message":
                vectorizer = sms_vectorizer
                model = sms_model
            else:
                vectorizer = email_vectorizer
                model = email_model

            message_vector = vectorizer.transform([message])
            prediction = str(model.predict(message_vector)[0]).strip().lower()

            if message_type == "SMS Message":
                if prediction == "spam":
                    st.error("🚨 Result: SPAM")
                    st.write("This SMS may be unwanted or deceptive.")
                else:
                    st.success("✅ Result: LIKELY LEGITIMATE")
                    st.write("The model did not classify this SMS as spam.")
            else:
                if prediction in ["phishing", "1", "true"]:
                    st.error("🚨 Result: POSSIBLE PHISHING")
                    st.write("This email may be attempting to deceive you.")
                else:
                    st.success("✅ Result: LIKELY LEGITIMATE")
                    st.write("The model did not classify this email as phishing.")

            st.caption(
                "Always verify suspicious senders and links independently. "
                "A legitimate result does not guarantee that a message is safe."
            )

except FileNotFoundError as error:
    st.error(f"Model file not found: {error}")
    st.write("Check that all four model files exist inside the model folder.")
except Exception as error:
    st.error(f"Something went wrong: {error}")
