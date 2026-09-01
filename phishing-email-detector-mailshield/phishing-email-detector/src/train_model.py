"""
MailShield - Model Training & Evaluation Module
-----------------------------------------------
Trains a Scikit-learn classification pipeline (TF-IDF + URL features + Classifier),
evaluates metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix),
generates visualization plots, and saves the trained model artifact.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import load_dataset, preprocess_dataframe
from feature_extraction import build_combined_feature_union


def evaluate_model_performance(y_true, y_pred, model_name="Logistic Regression"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "=" * 55)
    print(f"       MODEL EVALUATION: {model_name.upper()}")
    print("=" * 55)
    print(f"   Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"   Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"   Recall:    {rec:.4f} ({rec * 100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print("-" * 55)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Safe (0)", "Phishing (1)"]))
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_and_export_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "emails.csv")
    models_dir = os.path.join(base_dir, "models")
    model_output_path = os.path.join(models_dir, "phishing_model.pkl")
    
    os.makedirs(models_dir, exist_ok=True)
    
    df = load_dataset(dataset_path)
    df = preprocess_dataframe(df)
    
    X = df['text']
    y = df['label_num'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('features', build_combined_feature_union()),
        ('classifier', LogisticRegression(C=1.0, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    evaluate_model_performance(y_test, y_pred)
    joblib.dump(pipeline, model_output_path)
    print(f" Saved model to {model_output_path}")
    return pipeline


if __name__ == "__main__":
    train_and_export_model()
