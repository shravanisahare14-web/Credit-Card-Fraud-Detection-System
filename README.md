# 💳 Credit Card Fraud Detection System

## 📌 Overview
This project uses Machine Learning to detect fraudulent credit card transactions.

## 🚨 Problem Statement
Fraud transactions are extremely rare (~0.17%) but cause significant financial loss. Detecting them is challenging due to class imbalance.

## 💡 Solution
- Performed EDA to understand data distribution
- Used SMOTE to handle class imbalance
- Trained Random Forest model
- Evaluated using precision, recall, and F1-score

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- SMOTE
- Matplotlib, Seaborn

## 📊 Results
- Fraud Detection Recall: **84%**
- Precision: **89%**
- Balanced model performance

## 📷 Outputs

### Class Distribution
![Class Distribution](images/class_distribution.png)

### Correlation Heatmap
![Heatmap](images/correlation_heatmap.png)

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

## 📂 Dataset

Due to GitHub file size limitations, the dataset is not included.

You can download it from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file in:

data/creditcard.csv
