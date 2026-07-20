# =====================================================
# Bank Marketing Prediction using Decision Tree
# Author: Subhasisha Jena
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# -----------------------------------------------------
# Create output folder
# -----------------------------------------------------

os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("data/bank.csv", sep=";")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")

print(df.isnull().sum())

# -----------------------------------------------------
# Class Distribution
# -----------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x="y", data=df)

plt.title("Target Variable Distribution")

plt.savefig("outputs/class_distribution.png")

plt.close()

# -----------------------------------------------------
# Encode Categorical Columns
# -----------------------------------------------------

encoder = LabelEncoder()

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = encoder.fit_transform(df[col])

# -----------------------------------------------------
# Feature and Target
# -----------------------------------------------------

X = df.drop("y", axis=1)

y = df["y"]

# -----------------------------------------------------
# Train Test Split
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------

print("\nTraining Decision Tree...")

model = DecisionTreeClassifier(

    criterion="entropy",
    max_depth=6,
    random_state=42

)

model.fit(X_train, y_train)

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

y_pred = model.predict(X_test)

# -----------------------------------------------------
# Accuracy
# -----------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------

disp = ConfusionMatrixDisplay(

    confusion_matrix(y_test, y_pred)

)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.close()

# -----------------------------------------------------
# ROC Curve
# -----------------------------------------------------

RocCurveDisplay.from_estimator(

    model,
    X_test,
    y_test

)

plt.title("ROC Curve")

plt.savefig("outputs/roc_curve.png")

plt.close()

# -----------------------------------------------------
# Feature Importance
# -----------------------------------------------------

importance = pd.Series(

    model.feature_importances_,
    index=X.columns

).sort_values()

plt.figure(figsize=(8,7))

importance.plot(kind="barh")

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.close()

# -----------------------------------------------------
# Decision Tree
# -----------------------------------------------------

plt.figure(figsize=(20,10))

plot_tree(

    model,
    feature_names=X.columns,
    class_names=["No","Yes"],
    filled=True,
    fontsize=8

)

plt.savefig("outputs/decision_tree.png")

plt.close()

print("\nAll graphs saved inside outputs folder.")

print("\nProject Completed Successfully!")
