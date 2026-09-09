# 📩 Spam SMS Detector

A machine learning project that classifies SMS messages as **spam** or **ham** (legitimate) using TF-IDF features and **Multinomial Naive Bayes**, with a ready-to-use Streamlit web app for real-time inference.

This project follows a complete classification pipeline — data cleaning, exploratory analysis, text preprocessing, model benchmarking, ensemble experiments, final model selection, and deployment.

---

## 📌 Problem Statement

SMS spam is a common nuisance and potential security risk.  
The goal of this project is to build a binary classifier that can automatically detect spam messages from text content alone (using the classic SMS Spam Collection dataset) and package it into a simple interactive web application.

---

## 📁 Dataset

| Detail                    | Value                                      |
| ------------------------- | ------------------------------------------ |
| Source                    | SMS Spam Collection (UCI / Kaggle)         |
| File                      | `spam_data.csv` (latin-1 encoding)         |
| Total records             | 5,572                                      |
| After removing duplicates | ~5,169                                     |
| Missing values            | None                                       |
| Target variable           | `label` / `target` (`ham` / `spam`) → 0/1  |

**Class distribution (after deduplication):** Heavily imbalanced — majority of messages are `ham`.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – data handling
- **Matplotlib** & **Seaborn** – visualization
- **NLTK** – tokenization, stopwords, stemming
- **Scikit-learn** – TF-IDF vectorization, MultinomialNB, ensembles, metrics
- **Joblib** – model & vectorizer serialization
- **Streamlit** – interactive web app for predictions

---

## 🔄 Project Workflow

1. **Data Loading & Inspection**  
   Loaded `spam_data.csv` (latin-1 encoding), inspected shape, columns, and sample messages.

2. **Exploratory Data Analysis (EDA)**  
   - Class distribution  
   - Message length statistics  
   - Word clouds / most frequent terms for ham vs spam  

3. **Data Cleaning**  
   - Dropped unused columns (`Unnamed: 2/3/4`)  
   - Renamed columns appropriately  
   - Checked for missing values (none)  
   - Removed duplicate rows  

4. **Text Preprocessing**  
   Pipeline applied to every message:
   - Lowercasing  
   - Tokenization (`nltk.word_tokenize`)  
   - Keep only alphanumeric tokens  
   - Remove English stopwords and punctuation  
   - Porter stemming  

5. **Feature Engineering**  
   - TF-IDF vectorization (`TfidfVectorizer(max_features=3000)`) on the transformed text  

6. **Model Building & Benchmarking**  
   Multiple algorithms were trained and compared (SVC, KNeighbors, MultinomialNB, Decision Tree, Logistic Regression, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, XGBoost).

7. **Model Improvement & Ensembles**  
   - Soft Voting Classifier (SVC + MultinomialNB + Extra Trees)  
   - Stacking Classifier (same base models + Random Forest meta-learner)  

8. **Final Model Selection & Deployment**  
   - Selected **Multinomial Naive Bayes** for its excellent balance of high precision, good accuracy, and computational efficiency  
   - Saved vectorizer + model with joblib  
   - Built a Streamlit web app (`app.py`) for real-time predictions  

---

## 📊 Results

### Final Model — MultinomialNB

| Metric           | Score  |
| ---------------- | ------ |
| Accuracy         | 0.9710 |
| Precision (spam) | 1.0000 |

The model achieves **perfect precision** on the spam class on the held-out test set while maintaining high overall accuracy. This is desirable when false positives (flagging legitimate messages as spam) are costly.

### Other Notable Results (for reference)

| Model                        | Accuracy | Precision |
| ---------------------------- | -------- | --------- |
| BernoulliNB                  | 0.9836   | 0.9919    |
| Extra Trees                  | 0.9778   | 0.9675    |
| Random Forest                | 0.9768   | 0.9750    |
| SVC                          | 0.9758   | 0.9748    |
| **Voting Classifier (soft)** | **0.9807** | **0.9836** |
| Stacking Classifier          | 0.9787   | 0.9394    |
| **MultinomialNB (final)**    | **0.9710** | **1.0000** |

Although some ensembles and BernoulliNB scored higher accuracy, **MultinomialNB** was chosen as the production model because of its perfect precision, simplicity, speed, and lower resource requirements.

---

## ⚠️ Limitations

- Class imbalance is present; no oversampling, undersampling, or class-weight tuning was applied in the final model.
- Relies on classical bag-of-words (TF-IDF) features. Character n-grams, embeddings, or transformers can improve robustness further.
- Trained on a relatively clean public SMS dataset — may not fully generalize to modern multi-channel spam or adversarial text.
- No extensive hyperparameter search or cross-validation is reported beyond the single stratified train-test split used for final evaluation.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
streamlit==1.41.1
scikit-learn
joblib==1.4.2
nltk==3.9.1
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens in the browser. Paste any SMS message and click **Classify** to receive a spam/ham prediction with confidence score. An expander also shows the preprocessed text for debugging.

### 3. Notebook (training & exploration)

Open `SpamClassifier.ipynb` in Jupyter / Google Colab / VS Code and run all cells. It contains the full pipeline from data loading to model saving.

---

## 📂 Project Structure

```
.
├── app.py                 # Streamlit web application
├── model.joblib           # Trained MultinomialNB model
├── vectorizer.joblib      # Fitted TF-IDF vectorizer (max_features=3000)
├── SpamClassifier.ipynb   # Full training & analysis notebook
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md              # This file
```

---

## 🔮 Future Improvements

- Add class-weight handling, SMOTE, or focal loss to better address imbalance
- Experiment with character n-grams and richer text features
- Perform systematic cross-validation and hyperparameter tuning
- Evaluate on more recent or domain-specific spam datasets
- Add confidence calibration / threshold tuning controls in the Streamlit app
- Containerize the app (Docker) for easier deployment

---

## 📝 Conclusion

This project delivers a complete, end-to-end SMS spam detection system:

- A carefully preprocessed **TF-IDF + Multinomial Naive Bayes** classifier that achieves **~97.1% accuracy** and **perfect spam precision** on the test set.
- A clean Streamlit interface that lets anyone paste a message and receive an instant prediction with confidence.

The chosen model prioritizes high precision (very few false alarms on legitimate messages) while remaining lightweight and fast — a practical trade-off for a spam filter.

---

**Author:** Avinash Sharma
