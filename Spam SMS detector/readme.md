# Spam SMS Detector

A machine learning project that classifies SMS messages as **spam** or **ham** (legitimate) using TF-IDF features and Logistic Regression.

This project follows a classic classification pipeline — exploratory analysis, cleaning & preprocessing, feature engineering, model training, evaluation, and threshold analysis.

---

## 📌 Problem Statement

SMS spam is a common nuisance and potential security risk.  
The goal of this project is to build a binary classifier that can automatically detect spam messages from text content alone, using the classic SMS Spam Collection dataset.

**Note:** The current model is a **baseline Logistic Regression** model with TF-IDF features. It achieves strong overall accuracy but still shows a noticeable trade-off between precision and recall for the minority (spam) class (see Limitations section).

---

## 📁 Dataset

| Detail                    | Value                                      |
|---------------------------|--------------------------------------------|
| Source                    | SMS Spam Collection (UCI / Kaggle)         |
| File                      | `spam.csv`                                 |
| Total records             | 5,572                                      |
| After removing duplicates | 5,169                                      |
| Missing values            | None                                       |
| Target variable           | `label` (`ham` / `spam`) → encoded as 0/1  |

### Features
| Feature     | Description                                      |
|-------------|--------------------------------------------------|
| label       | Class label: `ham` (legitimate) or `spam`        |
| message     | Raw SMS text content                             |
| label_num   | Binary encoding of label (0 = ham, 1 = spam)     |
| msg_length  | Character length of the message (engineered)     |

**Class distribution (after deduplication):** Heavily imbalanced — majority of messages are `ham`.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – data handling
- **Matplotlib** & **Seaborn** – visualization
- **Scikit-learn** – TF-IDF vectorization, Logistic Regression, metrics

---

## 🔄 Project Workflow

1. **Data Loading & Inspection**  
   Loaded `spam.csv` (latin-1 encoding), inspected shape, columns, and sample messages.

2. **Exploratory Data Analysis (EDA)**  
   - Class distribution (`value_counts`)  
   - Message length statistics (mean ≈ 80 characters, max 910)  
   - Basic length comparison between ham and spam  

3. **Data Cleaning**  
   - Dropped unused columns (`Unnamed: 2/3/4`)  
   - Renamed columns to `label` and `message`  
   - Checked for missing values (none)  
   - Removed 403 duplicate rows  

4. **Feature Engineering**  
   - Encoded target: `label_num = 1` if spam else `0`  
   - Created `msg_length` feature  
   - Applied **TF-IDF** vectorization (`max_features=3000`, English stop-words) on the message text  

5. **Model Building**  
   - Stratified train-test split → 80% train / 20% test (`random_state=42`)  
   - Algorithm → **Logistic Regression** (`max_iter=1000`)  
   - Target → `label_num`  

6. **Evaluation & Threshold Analysis**  
   - Default threshold (0.5) metrics  
   - Threshold sweep (0.3, 0.5, 0.7, 0.9) examining Precision / Recall / FP / FN trade-offs  
   - Manual inspection of false positives and false negatives  

---

## 📊 Results

### Default Threshold (0.5)

| Metric              | Score   |
|---------------------|---------|
| Accuracy            | 0.961   |
| Spam Precision      | 0.989   |
| Spam Recall         | 0.702   |
| Spam F1-Score       | 0.82    |

**Confusion Matrix (test set, n=1034):**
```
[[902   1]   ← Ham correctly classified / False Positives
 [ 39  92]]  ← False Negatives / Spam correctly classified
```

### Threshold Sweep Summary

| Threshold | Precision | Recall | False Positives | False Negatives |
|-----------|-----------|--------|-----------------|-----------------|
| 0.3       | 0.949     | 0.847  | 6               | 20              |
| 0.5       | 0.989     | 0.702  | 1               | 39              |
| 0.7       | 1.000     | 0.435  | 0               | 74              |
| 0.9       | 1.000     | 0.084  | 0               | 120             |

**Interpretation:**  
At the default threshold the model is highly precise (very few legitimate messages flagged as spam) but misses ~30% of actual spam. Lowering the threshold improves recall at the cost of a few extra false positives.

---

## ⚠️ Limitations (Why it is not perfect)

- Only Logistic Regression + basic TF-IDF was used. More advanced approaches (n-grams, character-level features, transformers, or ensemble methods) usually perform better.
- Class imbalance is present; no oversampling, undersampling, or class-weight tuning was applied.
- No cross-validation or hyperparameter search was performed.
- Message length was engineered but not combined with TF-IDF in the final model.
- Residual / error analysis is limited to a handful of misclassified examples.
- The model is trained on a relatively clean, public SMS dataset and may not generalize well to modern multi-channel spam or adversarial text.

---

## 🚀 How to Run

### Requirements
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Steps
1. Keep the dataset file in the same folder as the notebook:
   ```
   spam.csv
   ```
2. Open `SpamClassifier.ipynb` in Jupyter Notebook / Google Colab / VS Code.
3. Run all cells from top to bottom.

---

## 📂 Project Structure

```
.
├── spam.csv                          #-----> Raw dataset
├── SpamClassifier.ipynb              #-----> Main notebook
└── README.md                         #-----> Project documentation
```

---

## 🔮 Future Improvements

- Experiment with class weights, SMOTE, or focal loss to handle imbalance
- Try multinomial Naive Bayes, Random Forest, or Gradient Boosting
- Add character n-grams or more sophisticated text features
- Perform proper cross-validation and hyperparameter tuning
- Evaluate on more recent or domain-specific spam datasets
- **Deploy the model** as a simple web app (Streamlit / Flask / FastAPI) so users can paste an SMS and get an instant spam/ham prediction — **deployment is planned for a later stage**

---

## 📝 Conclusion

This project successfully builds a complete text classification pipeline for SMS spam detection.  
The Logistic Regression + TF-IDF baseline achieves **~96% accuracy** and excellent precision, making it a solid starting point.  

However, the model is **not production-ready**. Significant gains in recall (and overall robustness) are possible with better imbalance handling, richer features, and stronger algorithms.  

**Deployment will be done later** once the model is further improved and packaged.

---

**Author:** Avinash Sharma
