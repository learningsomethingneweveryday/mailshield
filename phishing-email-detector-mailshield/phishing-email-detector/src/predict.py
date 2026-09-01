"""
MailShield - Prediction CLI Interface
-------------------------------------
Interactive command-line tool to classify incoming emails as Phishing or Safe
with confidence scoring and feature explainability.
"""

import os
import sys
import argparse
import joblib
from feature_extraction import extract_url_and_keyword_features


def load_trained_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "phishing_model.pkl")
    if not os.path.exists(model_path):
        from train_model import train_and_export_model
        return train_and_export_model()
    return joblib.load(model_path)


def predict_email(model, email_text: str):
    features = extract_url_and_keyword_features(email_text)
    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    
    prob_safe, prob_phishing = probabilities[0], probabilities[1]
    
    label = "PHISHING" if prediction == 1 else "SAFE"
    emoji = "🚨" if prediction == 1 else "✅"
    confidence = prob_phishing * 100 if prediction == 1 else prob_safe * 100
    
    return {
        "prediction": int(prediction),
        "label": label,
        "emoji": emoji,
        "confidence": confidence,
        "prob_phishing": prob_phishing,
        "prob_safe": prob_safe,
        "features": features
    }


def main():
    parser = argparse.ArgumentParser(description="MailShield Phishing Email Classifier")
    parser.add_argument("--text", "-t", type=str, help="Email text to classify.")
    args = parser.parse_args()
    
    model = load_trained_model()
    
    if args.text:
        res = predict_email(model, args.text)
        print(f"\nPrediction: {res['emoji']} {res['label']}")
        print(f"Confidence: {res['confidence']:.1f}%\n")
    else:
        print("--- MailShield CLI: Enter email below ---")
        user_input = input("Enter email: ")
        res = predict_email(model, user_input)
        print(f"\nPrediction: {res['emoji']} {res['label']}")
        print(f"Confidence: {res['confidence']:.1f}%")


if __name__ == "__main__":
    main()
