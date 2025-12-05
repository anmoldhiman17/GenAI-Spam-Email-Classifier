import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import io

# ---------------- Page config & styling ----------------
st.set_page_config(page_title="GenAI Spam Classifier", layout="wide", initial_sidebar_state="expanded")

# small CSS for modern look
st.markdown(
    """
    <style>
    .main > header {background: linear-gradient(90deg,#6a11cb,#2575fc); padding:16px 24px; color: white; border-radius:12px}
    .stApp { background: linear-gradient(180deg,#0f172a 0%, #071031 100%); color: #e6eef8 }
    .title {font-size:28px; font-weight:700}
    .subtitle {color: #dbeafe}
    .card {background: rgba(255,255,255,0.03); padding:14px; border-radius:12px}
    textarea {background: rgba(255,255,255,0.02)}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="title">📧 GenAI — Spam Email/SMS Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modern interface for quick dataset experiments, model training and message testing.</div>', unsafe_allow_html=True)
st.write("---")

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Configure & Upload")
    uploaded_file = st.file_uploader("Upload CSV (text + label)", type=["csv"]) 
    st.markdown("**Or use default SMS Spam dataset**")
    use_default = st.button("Load default SMS dataset")
    st.markdown("---")

    st.subheader("Preprocessing & Model")
    model_choice = st.selectbox("Classifier", ["MultinomialNB", "LogisticRegression"], index=0)
    stop_words = st.checkbox("Remove English stop words", value=True)
    ngram_min, ngram_max = st.select_slider("Ngram range (min,max)", options=[1,2,3], value=(1,1))
    test_size = st.slider("Test set proportion", 5, 50, 20)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    st.markdown("---")
    st.markdown("**Advanced**")
    show_class_report = st.checkbox("Show classification report (precision/recall/f1)", value=True)
    st.markdown("---")
    st.caption("Made with ❤️ — modify the code to suit your brand or colors")

# ---------------- Load dataset ----------------
@st.cache_data
def load_default_sms():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"]) 
    return df

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
elif use_default:
    df = load_default_sms()
    st.success("✅ Loaded default SMS dataset")
else:
    st.info("No dataset loaded — click 'Load default SMS dataset' in the sidebar or upload a CSV.")
    st.stop()

# show columns and allow user to select text/label columns
st.write("### Dataset preview")
st.write(df.head())
cols = list(df.columns)
text_col = st.selectbox("Select text column", options=cols, index=cols.index("text") if "text" in cols else 0)
label_col = st.selectbox("Select label column", options=cols, index=cols.index("label") if "label" in cols else min(1, len(cols)-1))

# Robust label interpretation
labels = df[label_col]

if pd.api.types.is_numeric_dtype(labels):
    y = labels.astype(int)
else:
    unique_vals = labels.dropna().unique()
    unique_vals_lower = [str(v).lower() for v in unique_vals]
    if set(unique_vals_lower) <= {"ham","spam"}:
        y = labels.map(lambda v: 0 if str(v).lower()=="ham" else 1).astype(int)
    elif set(unique_vals_lower) <= {"0","1"}:
        y = labels.astype(int)
    else:
        try:
            y = pd.to_numeric(labels)
            if y.isnull().any():
                raise ValueError("Label column conversion produced NaNs.")
            y = y.astype(int)
        except Exception:
            st.error(f"Could not interpret label column. Example labels: {unique_vals[:6]}")
            st.stop()

if y.isnull().any():
    st.error("Label column contains missing values after processing. Clean your data.")
    st.stop()

# Features
X = df[text_col].astype(str)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))

# Vectorize
vectorizer = CountVectorizer(stop_words="english" if stop_words else None, ngram_range=(ngram_min, ngram_max))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
if model_choice == "MultinomialNB":
    model = MultinomialNB()
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

# ---------------- Layout: Metrics + Confusion matrix ----------------
left, right = st.columns([2, 1])
with left:
    st.markdown("<div class='card'>",
                unsafe_allow_html=True)
    st.subheader("📊 Model performance")
    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.write(f"**Precision:** {precision:.2f}  •  **Recall:** {recall:.2f}  •  **F1-score:** {f1:.2f}")
    if show_class_report:
        st.text(classification_report(y_test, y_pred, digits=3))
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Ham','Spam']); ax.set_yticklabels(['Ham','Spam'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=14)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    st.pyplot(fig)

st.write("---")

# ---------------- Try your own message ----------------
st.subheader("🔎 Try your own message")
col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_area("Enter an email or SMS message here:", height=120)
with col2:
    predict_button = st.button("Classify message")

if predict_button:
    if user_input.strip():
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(user_vec)[0]
        label = "🚨 Spam" if int(prediction) == 1 else "✅ Ham (Not Spam)"
        st.markdown(f"### Result: {label}")
        if prob is not None:
            st.write(f"Confidence — Ham: {prob[0]*100:.2f}%  •  Spam: {prob[1]*100:.2f}%")
    else:
        st.warning("Please enter a message first.")

# ---------------- Show random examples ----------------
st.write("---")
st.subheader("Sample messages from dataset")
example_tab1, example_tab2 = st.tabs(["Ham examples","Spam examples"])
with example_tab1:
    ham_examples = df.loc[y==0, text_col].sample(min(6, (y==0).sum()), random_state=1).tolist()
    for i, ex in enumerate(ham_examples, 1):
        st.markdown(f"**{i}.** {ex}")
with example_tab2:
    spam_examples = df.loc[y==1, text_col].sample(min(6, (y==1).sum()), random_state=1).tolist()
    for i, ex in enumerate(spam_examples, 1):
        st.markdown(f"**{i}.** {ex}")

# ---------------- Optional: Export vectorizer ----------------
buffer = io.BytesIO()
# lightweight export: column names + sample
export_df = pd.DataFrame({"text_sample": X.head(200), "label_sample": y.head(200)})
csv_bytes = export_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download sample CSV", data=csv_bytes, file_name="sample_export.csv", mime="text/csv")

st.caption("Tip: To make it look even more 'app-like' you can host as Streamlit Cloud and add a custom logo and favicon.")
