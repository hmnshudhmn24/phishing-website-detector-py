# Phishing Website Detector

## Overview
Phishing Website Detector is a machine learning-based tool that detects phishing websites by analyzing URL features.

## Features
- Extracts key URL features to identify phishing sites
- Uses a trained machine learning model for classification
- Simple and lightweight implementation
- Can analyze any given website URL

## Installation
Install required dependencies:
```sh
pip install pandas numpy scikit-learn joblib requests
```

## Usage
1. Train the model:
```sh
python phishing_detector.py
```
2. Predict if a website is phishing:
Modify the `test_url` in the script and run:
```sh
python phishing_detector.py
```
