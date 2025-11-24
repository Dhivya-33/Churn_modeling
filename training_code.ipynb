import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("Churn_Modelling.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

print("\nExited value counts:")
print(df['Exited'].value_counts())

print("\nChurn by Gender:")
print(df.groupby('Gender')['Exited'].mean())

print("\nChurn by Geography:")
print(df.groupby('Geography')['Exited'].mean())

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

X = df.drop('Exited', axis=1)
y = df['Exited']

# ------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


joblib.dump(scaler, "scaler.pkl")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")

print("\nModel & Scaler Saved Successfully!")
