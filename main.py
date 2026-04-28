# ==============================
# CREDIT CARD FRAUD DETECTION
# ==============================

# -------- IMPORTS --------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


# -------- LOAD DATA --------
print("🔹 Loading dataset...")
df = pd.read_csv('data/creditcard.csv')

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)


# -------- CLASS DISTRIBUTION --------
print("\nClass Distribution:")
print(df['Class'].value_counts())


# -------- EDA: CLASS DISTRIBUTION PLOT --------
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.savefig("images/class_distribution.png")
plt.show()


# -------- EDA: CORRELATION HEATMAP --------
plt.figure(figsize=(12,8))
corr = df.corr()

sns.heatmap(corr, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()


# -------- FEATURE ENGINEERING --------
print("\n🔹 Preparing features...")

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# -------- HANDLE IMBALANCE (SMOTE) --------
print("\n🔹 Applying SMOTE...")

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train_res))


# -------- MODEL TRAINING --------
print("\n🔹 Training model...")

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_res, y_train_res)


# -------- SAVE MODEL --------
joblib.dump(model, 'models/fraud_model.pkl')
print("✅ Model saved in models/fraud_model.pkl")


# -------- PREDICTIONS --------
print("\n🔹 Making predictions...")

y_pred = model.predict(X_test)


# -------- EVALUATION --------
print("\n🔹 Model Evaluation:\n")

report = classification_report(y_test, y_pred)
print(report)

# Save report
with open("outputs/report.txt", "w") as f:
    f.write(report)


# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Fraud'],
            yticklabels=['Normal','Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()


# -------- SIMULATION (PREDICTION SYSTEM) --------
print("\n🔹 Running fraud detection simulation...")

loaded_model = joblib.load('models/fraud_model.pkl')

sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)

if prediction[0] == 1:
    print("🚨 FRAUD ALERT! Transaction is suspicious.")
else:
    print("✅ Normal Transaction.")


print("\n🎯 Project Execution Completed Successfully!")