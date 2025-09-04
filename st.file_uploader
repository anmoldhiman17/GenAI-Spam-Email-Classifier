import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

st.title("📧 Spam Email/SMS Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
else:
    st.info("No file uploaded. Using default SMS Spam Collection dataset...")
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])

st.write("### Dataset Preview")
st.write(df.head())

# Preprocessing
X = df['text']
y = df['label'].map({'ham':0, 'spam':1})  # Convert ham/spam to 0/1

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
