# 📩 SMS Spam Classifier

An end-to-end Machine Learning project that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing and classical ML techniques.

The project started as a model evaluation experiment and was later rebuilt into a complete application with a **trained model, saved artifacts, GitHub repository, and live Streamlit deployment**.

## 🚀 Live Demo

👉 **[Try the SMS Spam Classifier](https://spamclassifi.streamlit.app/)**

Enter any SMS message and the application will predict whether it is **Spam** or **Not Spam**.

## 📌 Project Overview

The goal of this project is to build a practical spam detection system while understanding an important ML concept:

> **A model can have high accuracy and still make the wrong mistakes.**

During the project, I explored model performance using metrics such as accuracy and precision and compared multiple classification algorithms.

The final version was packaged into a usable Streamlit application.

## 📊 Dataset

The project uses the **SMS Spam Collection dataset**, containing **5,572 SMS messages** labeled as either:

- `ham` → legitimate message
- `spam` → spam message

The dataset was cleaned before model development. The original data contained additional unused columns, which were removed, and duplicate rows were checked and handled. 
## 🔄 ML Workflow

```text
SMS Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Model Training
     ↓
Model Comparison
     ↓
Final Model
     ↓
Save Model + Vectorizer
     ↓
Streamlit Deployment
```

The notebook follows this overall workflow from data preparation through model selection and deployment.

## 🧠 Models Explored

Several classical ML algorithms were benchmarked, including:

- Logistic Regression
- Support Vector Classifier
- Decision Tree
- K-Nearest Neighbors
- Multinomial Naive Bayes
- Random Forest
- AdaBoost
- Bagging
- Extra Trees
- Gradient Boosting
- XGBoost

The comparison focused on both **accuracy and precision**, with precision being particularly useful for understanding false-positive behavior in spam detection.

### 📈 Model Comparison

| Model | Accuracy | Precision |
|---|---:|---:|
| Multinomial Naive Bayes | 97.10% | 100% |
| Random Forest | 97.68% | 97.50% |
| SVC | 97.58% | 97.48% |
| Extra Trees | 97.78% | 96.75% |
| Logistic Regression | 95.65% | 96.97% |
| XGBoost | 96.91% | 94.92% |

These are the benchmark results recorded in the notebook.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- TF-IDF
- Joblib
- Streamlit

## 💾 Model Deployment

The trained TF-IDF vectorizer and final model are saved using `joblib` so they can be loaded by the deployed application.

The application is deployed using **Streamlit** and can be tested directly from the browser.

## 🖥️ Try It Yourself

### 1. Clone the repository

```bash
git clone https://github.com/Avinash01715/ds-ml-projects.git
cd ds-ml-projects/2.%20Spam%20SMS%20detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

## 💡 What I Learned

This project helped me understand that building an ML model is not just about achieving the highest accuracy.

I learned about:

- Text preprocessing
- TF-IDF feature extraction
- Comparing multiple ML algorithms
- Precision vs. accuracy
- False positives and false negatives
- Model persistence using Joblib
- Turning an ML notebook into an application
- Deploying an ML project with Streamlit

## 🔮 Future Improvements

- Better handling of false negatives
- Threshold tuning and probability calibration
- Testing on newer spam/scam messages
- Adding more recent SMS datasets
- Improving the UI
- Exploring modern NLP models

## 👨‍💻 Author

**Avinash Sharma**

Building and learning through practical Data Science & Machine Learning projects.

### 🔗 Links

- 💻 [GitHub Repository](https://github.com/Avinash01715/ds-ml-projects/tree/main/2.%20Spam%20SMS%20detector)
- 🚀 [Live Demo](https://spamclassifi.streamlit.app/)