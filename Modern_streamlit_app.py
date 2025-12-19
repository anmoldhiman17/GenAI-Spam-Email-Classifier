import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="📧 AI Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
    }
    
    .main {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 5px;
        color: #047857;
    }
    
    .error-box {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 5px;
        color: #991b1b;
    }
    
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 5px;
        color: #92400e;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "test_data" not in st.session_state:
    st.session_state.test_data = None

# Header
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>📧 AI Spam Email/SMS Classifier</h1>", 
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Advanced ML-powered spam detection with real-time classification</p>", 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)
    max_features = st.slider("Max Features (Vectorizer)", 100, 5000, 1000, step=100)
    
    st.markdown("---")
    st.markdown("### 📊 Data Options")
    data_source = st.radio("Choose Data Source:", ["Upload CSV", "Use Default Dataset"])

# Main Layout
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train Model", "🔍 Classify Messages", "📊 Analytics", "📈 Detailed Report"])

# ==================== TAB 1: TRAIN MODEL ====================
with tab1:
    st.markdown("### 📁 Upload & Train Your Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.test_data = df
                    
                    st.success("✅ Dataset loaded successfully!")
                    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
                        
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        else:
            st.info("📥 Loading default SMS Spam Collection dataset...")
            try:
                url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
                df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])
                st.session_state.test_data = df
                st.success("✅ Default dataset loaded!")
            except:
                st.error("❌ Could not load default dataset. Please upload your own.")
    
    with col2:
        if st.session_state.test_data is not None:
            st.metric("Dataset Size", f"{len(st.session_state.test_data)} messages")
    
    # Dataset Preview
    if st.session_state.test_data is not None:
        st.markdown("#### Dataset Preview")
        st.dataframe(st.session_state.test_data.head(10), use_container_width=True)
        
        # Data Statistics
        st.markdown("#### 📊 Data Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.test_data))
        with col2:
            unique_labels = st.session_state.test_data['label'].unique()
            st.metric("Unique Labels", len(unique_labels))
        with col3:
            st.metric("Text Column", "text")
        
        # Train Model Button
        if st.button("🚀 Train Model", use_container_width=True, key="train_btn"):
            with st.spinner("⏳ Training model..."):
                try:
                    df = st.session_state.test_data
                    
                    # Determine text and label columns
                    if "text" in df.columns:
                        X = df["text"].astype(str)
                    elif "origin" in df.columns:
                        X = df["origin"].astype(str)
                    else:
                        st.error("❌ No valid text column found!")
                        st.stop()
                    
                    labels = df['label']
                    
                    # Label Processing
                    if pd.api.types.is_numeric_dtype(labels):
                        y = labels.astype(int)
                    else:
                        unique_vals_lower = [str(v).lower() for v in labels.dropna().unique()]
                        if set(unique_vals_lower) <= {"ham", "spam"}:
                            y = labels.map(lambda v: 0 if str(v).lower() == "ham" else 1).astype(int)
                        elif set(unique_vals_lower) <= {"0", "1"}:
                            y = labels.astype(int)
                        else:
                            y = pd.to_numeric(labels, errors='coerce').astype(int)
                    
                    if y.isnull().any():
                        st.error("❌ Label column contains missing values!")
                        st.stop()
                    
                    # Train-Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Vectorization
                    vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                    
                    # Model Training
                    model = MultinomialNB()
                    model.fit(X_train_vec, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test_vec)
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Store in session
                    st.session_state.model = model
                    st.session_state.vectorizer = vectorizer
                    st.session_state.model_trained = True
                    st.session_state.predictions = {
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'X_test': X_test.values
                    }
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    with col2:
                        st.metric("Precision", f"{precision*100:.2f}%")
                    with col3:
                        st.metric("Recall", f"{recall*100:.2f}%")
                    with col4:
                        st.metric("F1-Score", f"{f1*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")

# ==================== TAB 2: CLASSIFY MESSAGES ====================
with tab2:
    st.markdown("### 🔍 Test Message Classification")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Train Model' tab!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter an email or SMS message:",
                placeholder="Type or paste your message here...",
                height=150,
                key="message_input"
            )
        
        with col2:
            st.markdown("#### Tips:")
            st.markdown("- Be specific")
            st.markdown("- Include common spam phrases")
            st.markdown("- Use Ctrl+Enter to classify")
        
        if st.button("✨ Classify Message", use_container_width=True, key="classify_btn"):
            if user_input.strip():
                try:
                    user_vec = st.session_state.vectorizer.transform([user_input])
                    prediction = st.session_state.model.predict(user_vec)[0]
                    probability = st.session_state.model.predict_proba(user_vec)[0]
                    
                    if prediction == 1:
                        st.markdown("""
                        <div style='background-color: #fee2e2; border-left: 4px solid #ef4444; padding: 20px; border-radius: 5px;'>
                            <h3 style='color: #991b1b; margin: 0;'>🚨 SPAM DETECTED</h3>
                            <p style='color: #7f1d1d; margin-top: 10px;'>This message is classified as SPAM with <strong>{:.1f}%</strong> confidence.</p>
                        </div>
                        """.format(probability[1] * 100), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background-color: #d1fae5; border-left: 4px solid #10b981; padding: 20px; border-radius: 5px;'>
                            <h3 style='color: #047857; margin: 0;'>✅ LEGITIMATE</h3>
                            <p style='color: #065f46; margin-top: 10px;'>This message is legitimate with <strong>{:.1f}%</strong> confidence.</p>
                        </div>
                        """.format(probability[0] * 100), unsafe_allow_html=True)
                    
                    # Confidence Bar
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Spam Confidence:**")
                        st.progress(probability[1])
                    with col2:
                        st.write("**Ham Confidence:**")
                        st.progress(probability[0])
                
                except Exception as e:
                    st.error(f"❌ Error during classification: {str(e)}")
            else:
                st.warning("⚠️ Please enter a message to classify!")

# ==================== TAB 3: ANALYTICS ====================
with tab3:
    st.markdown("### 📊 Model Analytics & Insights")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first!")
    else:
        pred_data = st.session_state.predictions
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Accuracy", f"{pred_data['accuracy']*100:.2f}%")
        with col2:
            st.metric("🔍 Precision", f"{pred_data['precision']*100:.2f}%")
        with col3:
            st.metric("🎪 Recall", f"{pred_data['recall']*100:.2f}%")
        with col4:
            st.metric("⚡ F1-Score", f"{pred_data['f1']*100:.2f}%")
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(pred_data['y_test'], pred_data['y_pred'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Ham', 'Spam'], 
                       yticklabels=['Ham', 'Spam'],
                       cbar=False)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Class Distribution")
            class_dist = pd.DataFrame({
                'Actual': ['Ham', 'Spam'],
                'Count': [
                    (pred_data['y_test'] == 0).sum(),
                    (pred_data['y_test'] == 1).sum()
                ]
            })
            
            fig = px.pie(class_dist, values='Count', names='Actual',
                        color_discrete_map={'Ham': '#10b981', 'Spam': '#ef4444'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Classification Report
        st.markdown("#### Detailed Classification Report")
        from sklearn.metrics import classification_report
        report_dict = classification_report(pred_data['y_test'], pred_data['y_pred'], 
                                          output_dict=True, 
                                          target_names=['Ham', 'Spam'])
        
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

# ==================== TAB 4: DETAILED REPORT ====================
with tab4:
    st.markdown("### 📈 Comprehensive Analysis Report")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first!")
    else:
        pred_data = st.session_state.predictions
        
        # Summary
        st.markdown("#### 📋 Model Training Summary")
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            - **Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - **Algorithm:** Multinomial Naive Bayes
            - **Vectorizer:** Count Vectorizer (English stop words removed)
            - **Test Size:** {test_size*100:.0f}%
            - **Max Features:** {max_features}
            """)
        
        with summary_col2:
            st.markdown(f"""
            - **Total Test Samples:** {len(pred_data['y_test'])}
            - **True Positives (TP):** {((pred_data['y_pred'] == 1) & (pred_data['y_test'] == 1)).sum()}
            - **True Negatives (TN):** {((pred_data['y_pred'] == 0) & (pred_data['y_test'] == 0)).sum()}
            - **False Positives (FP):** {((pred_data['y_pred'] == 1) & (pred_data['y_test'] == 0)).sum()}
            - **False Negatives (FN):** {((pred_data['y_pred'] == 0) & (pred_data['y_test'] == 1)).sum()}
            """)
        
        # Performance Metrics Comparison
        st.markdown("#### 🎯 Performance Metrics Breakdown")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [
                f"{pred_data['accuracy']*100:.2f}%",
                f"{pred_data['precision']*100:.2f}%",
                f"{pred_data['recall']*100:.2f}%",
                f"{pred_data['f1']*100:.2f}%"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Test Sample Results
        st.markdown("#### 📊 Test Sample Classifications (First 20)")
        sample_results = pd.DataFrame({
            'Message': pred_data['X_test'][:20],
            'Actual Label': ['Spam' if x == 1 else 'Ham' for x in pred_data['y_test'][:20]],
            'Predicted Label': ['Spam' if x == 1 else 'Ham' for x in pred_data['y_pred'][:20]],
            'Correct': ['✅' if pred_data['y_test'].iloc[i] == pred_data['y_pred'][i] else '❌' for i in range(min(20, len(pred_data['y_test'])))]
        })
        st.dataframe(sample_results, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🛡️ <strong>Privacy First</strong> - All processing happens locally</p>
    <p>Built with ❤️ using Scikit-learn & Streamlit | AI/ML Powered Spam Detection</p>
    <p><small>© 2025 AI Spam Classifier | For Educational & Professional Use</small></p>
</div>
""", unsafe_allow_html=True)
