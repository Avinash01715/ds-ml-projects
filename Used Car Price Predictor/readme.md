# Used Car Price Predictor

A machine learning project that predicts the selling price of used cars based on features such as brand, age, kilometers driven, fuel type, transmission, seller type, and ownership history.

This project follows a classic regression pipeline — exploratory analysis, feature engineering, model training, and evaluation.

---

## 📌 Problem Statement

Buying or selling a used car involves uncertainty around fair market value.  
The goal of this project is to build a regression model that can estimate the **selling price** of a used car using publicly available CarDekho listings.

**Note:** The current model is a **baseline Linear Regression** model. It is **not perfect**. It captures major trends but still has significant room for improvement (see Limitations section).

---

## 📁 Dataset

| Detail              | Value                                      |
|---------------------|--------------------------------------------|
| Source              | CarDekho used car listings                 |
| File                | `CAR DETAILS FROM CAR DEKHO.csv`           |
| Total records       | 4,340                                      |
| After removing duplicates | ~3,500+                              |
| Missing values      | None                                       |
| Target variable     | `selling_price` (in INR)                   |

### Features
| Feature         | Description                                      |
|-----------------|--------------------------------------------------|
| name            | Full car model name (used to extract brand)      |
| year            | Year of manufacture                              |
| selling_price   | Listed selling price (target)                    |
| km_driven       | Total kilometers driven                          |
| fuel            | Petrol / Diesel / CNG / LPG / Electric           |
| seller_type     | Individual / Dealer / Trustmark Dealer           |
| transmission    | Manual / Automatic                               |
| owner           | First Owner / Second Owner / Third Owner / etc.  |

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** & **NumPy** – data handling
- **Matplotlib** & **Seaborn** – visualization
- **Scikit-learn** – preprocessing & modeling
- **Scipy** – statistical tests (Pearson correlation)

---

## 🔄 Project Workflow

1. **Data Loading & Inspection**  
   Loaded the CSV, checked shape, data types, missing values, and basic statistics.

2. **Exploratory Data Analysis (EDA)**  
   - Histograms and KDE plots for numeric features  
   - Count plots for categorical features  
   - Box plots for outlier inspection  
   - Correlation heatmap  

3. **Data Cleaning**  
   - Removed duplicate rows  
   - No missing values to handle  

4. **Feature Engineering**  
   - Extracted **brand** from the car name  
   - Grouped rare brands into “Other”  
   - Created **age** of the car (`2026 - year`)  
   - Applied **log transformation** on `km_driven` and `selling_price` (both were right-skewed)  
   - Created `km_per_year` as a usage intensity feature  
   - One-hot encoded categorical variables (`fuel`, `seller_type`, `transmission`, `owner`, `brand`)  
   - Scaled selected continuous features using `StandardScaler`  

5. **Feature Selection**  
   - Calculated Pearson correlation with the log-transformed target  
   - Performed VIF analysis to check multicollinearity  
   - Selected the final set of 22 features  

6. **Model Building**  
   - Train-test split → 80% train / 20% test (`random_state=42`)  
   - Algorithm → **Linear Regression**  
   - Target → `selling_price_log`  

7. **Evaluation**  
   - R² Score and Adjusted R²  
   - Converted predictions back to original price scale using `np.expm1`

---

## 📊 Results

| Metric          | Score   |
|-----------------|---------|
| R² Score        | 0.736   |
| Adjusted R²     | 0.727   |

**Interpretation:**  
The model explains roughly **73–74%** of the variance in log-transformed selling prices.  
This is a reasonable **baseline** result for a simple linear model, but it is **not highly accurate**.

---

## ⚠️ Limitations (Why it is not perfect)

- Only Linear Regression was used. Non-linear models (Random Forest, XGBoost, etc.) usually perform better on this type of data.
- Important real-world factors are missing (condition of the car, location, service history, exact variant, etc.).
- Brand information is coarse (only top brands kept).
- Residual analysis and error distribution were not deeply explored.
- No hyperparameter tuning or cross-validation was performed.
- Prediction intervals / uncertainty estimates are not provided.

---

## 🚀 How to Run

### Requirements
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

### Steps
1. Keep the dataset file in the same folder as the notebook:
   ```
   CAR DETAILS FROM CAR DEKHO.csv
   ```
2. Open `CarDekho_Used_Car_Price_Predictor.ipynb` in Jupyter Notebook / Google Colab / VS Code.
3. Run all cells from top to bottom.

---

## 📂 Project Structure

```
.
├── CAR DETAILS FROM CAR DEKHO.csv               #-----> Raw dataset
├── CarDekho_Used_Car_Price_Predictor.ipynb      #-----> Main notebook
└── README.md                                    #-----> Project documentation
```

---

## 🔮 Future Improvements

- Train tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Perform proper cross-validation and hyperparameter tuning
- Add more domain features if additional data becomes available
- Build a simple Streamlit / Flask web app for interactive predictions
- Analyze residuals and improve handling of high-value / luxury cars

---

## 📝 Conclusion

This project successfully builds a complete regression pipeline for used car price prediction.  
The Linear Regression model achieves a moderate R² of ~0.73 and serves as a solid **baseline**.  

However, the model is **not production-ready**. Significant improvement is possible by using more powerful algorithms and richer features.

---

**Author:** Aviansh Sharma

