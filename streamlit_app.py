import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

st.title("📧 Spam Email/SMS Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

    # text column: prefer 'text', otherwise 'origin'
    if "text" in df.columns:
        X = df["text"].astype(str)
    elif "origin" in df.columns:
        X = df["origin"].astype(str)
    else:
        st.error("❌ ERROR: No valid text column found! Expected 'text' or 'origin'.")
        st.stop()

else:
    st.info("No file uploaded. Using default SMS Spam Collection dataset...")
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])
    X = df["text"].astype(str)

st.write("### Dataset Preview")
st.write(df.head())

# ----- Robust label handling -----
labels = df['label']

# If numeric (ints/floats) — assume 0/1 already
if pd.api.types.is_numeric_dtype(labels):
    y = labels.astype(int)
else:
    # common string cases: 'ham'/'spam' or '0'/'1' as strings
    unique_vals = labels.dropna().unique()
    unique_vals_lower = [str(v).lower() for v in unique_vals]

    if set(unique_vals_lower) <= {"ham","spam"}:
        y = labels.map(lambda v: 0 if str(v).lower()=="ham" else 1).astype(int)
    elif set(unique_vals_lower) <= {"0","1"}:
        y = labels.astype(int)
    else:
        # try to coerce to numeric as last resort
        try:
            y = pd.to_numeric(labels)
            if y.isnull().any():
                raise ValueError("Label column conversion produced NaNs.")
            y = y.astype(int)
        except Exception as e:
            st.error(f"Could not interpret label column. Unique label examples: {unique_vals[:5]}.")
            st.stop()

# final safety check
if y.isnull().any():
    st.error("Label column contains missing values after processing. Clean your data.")
    st.stop()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

st.write("### 📊 Model Performance")
st.write(f"**Accuracy:** {acc*100:.2f}%")
st.text(classification_report(y_test, y_pred))

# User Input for Prediction
st.write("### 🔎 Try Your Own Message")
user_input = st.text_area("Enter an email or SMS message here:")
if st.button("Classify"):
    if user_input.strip():
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        label = "🚨 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
        st.subheader(label)
    else:
        st.warning("Please enter a message first.")
