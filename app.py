import streamlit as st
import joblib

# Load trained model
model = joblib.load("spam_model.pkl")

# Streamlit UI
st.set_page_config(page_title="GenAI Spam Email Classifier", page_icon="📧")

st.title("📧 GenAI Spam Email Classifier")
st.write("Detect whether an email is **Spam or Ham (Safe)** using a trained ML model.")

# Input box
email_text = st.text_area("✍️ Enter email text here:")

# Predict button
if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some text before classifying.")
    else:
        prediction = model.predict([email_text])[0]
        probability = model.predict_proba([email_text])[0][1]

        if prediction == 1:
            st.error(f"🚨 Spam Detected! (Confidence: {probability:.2f})")
        else:
            st.success(f"✅ Safe Email (Confidence: {1 - probability:.2f})")
