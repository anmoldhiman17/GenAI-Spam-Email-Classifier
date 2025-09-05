📧 GenAI Spam Email Classifier

A state-of-the-art spam email classification pipeline powered by Generative AI 🧠, NLP 🔤, and Machine Learning 🤖.
This project not only classifies emails as Spam 📮 or Ham 📬, but also tackles data imbalance using synthetic text generation.

✨ Features

✅ Exploratory Data Analysis (EDA) — distribution, text length, top spam words 📊
✅ Data Augmentation — template-based synthetic text & optional GPT/T5 for richer spam generation ✨
✅ Preprocessing — cleaning, stopword removal, lemmatization 🔎
✅ Models — Logistic Regression, XGBoost, and optional DistilBERT 🚀
✅ Evaluation — Precision, Recall, F1-score, ROC & Confusion Matrices 📈
✅ Explainability — top n-grams contributing to spam detection 🔍

📂 Dataset

The dataset is provided in compressed .gz format due to GitHub file size limits.

To extract locally:

gunzip email_origin.csv.gz
gunzip email_text.csv.gz

Or load directly in Python:

import pandas as pd

df1 = pd.read_csv("email_origin.csv.gz", compression="gzip")

df2 = pd.read_csv("email_text.csv.gz", compression="gzip")

👉 Full dataset and Project Run Demo is also available on Google Drive for convenience:

👉 [Open in Google Drive](https://drive.google.com/drive/folders/1_jocUwwgwQqdzY-07mw8s6Zuv5UKsayH?usp=sharing)


📑 Report & 🎥 Presentation

For those who want to dive deeper into the project:

📘## 📑 Report & 🎥 Presentation  

For those who want to **dive deeper into the project**:  

- 📘 **Detailed Project Report — complete methodology, results, and insights**  
  👉 [Tap to View](https://drive.google.com/file/d/16cOAUXBwmDcFbgWjgYaqCXBHibuOvEKp/view?usp=sharing)  

- 🎤 **Presentation Slides — a concise, visual summary of the entire workflow**  
  👉 [Tap to View](https://docs.google.com/presentation/d/1OS-OxRGxwxfjs_SRfikOYtgjYWsMAvPI/edit?usp=sharing&ouid=115648615652004455704&rtpof=true&sd=true)  

 — a concise, visual summary of the entire workflow.

👉 Both open directly in Google Drive so you can view them instantly.

👉 Both are available in this repo under /report and /slides folders. Perfect for reviewers, interviewers, or anyone who wants the “big picture” + technical depth!

🛠️ Tech Stack

Python 🐍

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

NLP: spaCy, HuggingFace Transformers (GPT-2, DistilBERT)

ML Models: Logistic Regression, Random Forest, XGBoost

Visualization: matplotlib, seaborn

🚀 Project Workflow

EDA.ipynb → Dataset exploration 📊

Text_Generation.ipynb → Generate synthetic spam ✨

Model_Training.ipynb → Train & compare models 🤖

Visualization.ipynb → Confusion Matrix, ROC Curve 📈

📊 Results (Sample)

| Model                   | Precision | Recall | F1-score | ROC AUC |
| ----------------------- | --------- | ------ | -------- | ------- |
| Logistic Regression     | 0.93      | 0.88   | 0.90     | 0.95    |
| XGBoost                 | 0.95      | 0.90   | 0.92     | 0.97    |
| DistilBERT (Fine-tuned) | 0.97      | 0.94   | 0.95     | 0.98    |

📌 Future Improvements

Expand augmentation with advanced LLMs (T5, GPT-4)

Deploy as a Streamlit web app for real-time spam detection

Add SHAP/LIME explainability for model transparency

🔎 Conclusion

This project demonstrates how Generative AI can overcome class imbalance in real-world datasets like spam detection. 
By blending synthetic text with classical ML and modern transformers, the pipeline achieves strong performance while maintaining scalability and transparency.

Spam doesn’t stand a chance 💥 — built with love & logic by Anmol Dhiman.

