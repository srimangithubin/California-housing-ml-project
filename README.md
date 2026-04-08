# California-housing-ml-project
# 🏠 California Housing Price Prediction — End-to-End ML Project

This project builds a robust regression pipeline to predict house prices using the California Housing dataset.  
The focus is not just model performance, but **correct machine learning methodology**:
data validation, proper splitting, avoiding data leakage, feature engineering, and model evaluation.

---

## 📌 Dataset
California Housing dataset from sklearn.

Target: `MedHouseVal` (Median House Value)

---

## 🚨 Key Learning: Why Naive Train-Test Split Fails

The dataset is **geographically ordered**, not random.

A simple random split caused:
- Distribution shift between train and test
- Biased evaluation

This was diagnosed using distribution plots and corrected using **stratified sampling based on median income**.

---

## 🧠 ML Concepts Demonstrated

- Stratified Train–Test Split (using income bins)
- Avoiding Data Leakage (split before scaling/engineering)
- Feature Engineering:
  - RoomsPerPerson
  - BedroomsRatio
  - BedroomsPerPerson
- Standardization and its effect on Gradient Descent
- Cross Validation
- Regularization using ElasticNet
- Proper Regression Diagnostics (Predicted vs Actual plots)

---

## ⚙️ Model Used

ElasticNet Regression (tuned)

| Metric | Score |
|-------|-------|
| CV R² | 0.656 |
| Test R² | 0.652 |
| RMSE | 0.64 |

These are **expected ceiling results for linear models** on this dataset.

---

## 📊 Diagnostic Plots

- Train vs Test distribution check
- Predicted vs Actual (Train/Test)
- Evidence of price cap at 5.0
- Model generalization without overfitting

---

## 🗂️ Features Used





---

## 🛡️ Data Leakage Prevention

- Split performed before scaling and engineering
- Stratified sampling to maintain real-world distribution

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook California_Housing_project.ipynb
