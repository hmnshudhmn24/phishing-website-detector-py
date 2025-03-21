import pandas as pd
import numpy as np
import re
import requests
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to extract features from URLs
def extract_features(url):
    features = []
    parsed_url = urlparse(url)

    # Feature 1: Length of URL
    features.append(len(url))

    # Feature 2: Presence of "@" symbol
    features.append(1 if "@" in url else 0)

    # Feature 3: Presence of hyphen in domain
    features.append(1 if "-" in parsed_url.netloc else 0)

    # Feature 4: Presence of multiple subdomains
    features.append(1 if parsed_url.netloc.count(".") > 2 else 0)

    # Feature 5: Uses HTTPS
    features.append(1 if parsed_url.scheme == "https" else 0)

    # Feature 6: Presence of IP address in URL
    features.append(1 if re.match(r"(\d{1,3}\.){3}\d{1,3}", parsed_url.netloc) else 0)

    # Feature 7: Check if domain is short
    features.append(1 if len(parsed_url.netloc) < 10 else 0)

    return np.array(features)

# Load dataset (Replace with an actual dataset for training)
def load_data():
    data = [
        ["https://securebank.com/login", 0],
        ["http://phishingsite.com", 1],
        ["http://192.168.1.1/login", 1],
        ["https://legitimate-site.com/home", 0],
        ["http://fake-site.net?secure=login", 1],
    ]
    df = pd.DataFrame(data, columns=["url", "label"])
    return df

# Train the model
def train_model():
    df = load_data()
    df["features"] = df["url"].apply(lambda x: extract_features(x).tolist())
    X = np.array(df["features"].tolist())
    y = np.array(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, "phishing_detector.pkl")
    print("Model saved as phishing_detector.pkl")

# Predict phishing or legitimate
def predict(url):
    model = joblib.load("phishing_detector.pkl")
    features = extract_features(url).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

# Example Usage
if __name__ == "__main__":
    train_model()
    test_url = "http://example.com/login"
    print(f"URL: {test_url} -> {predict(test_url)}")
