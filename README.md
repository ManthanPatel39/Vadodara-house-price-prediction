# Vadodara-house-price-prediction
Vadodara House Price Predictor: A Flask-based web app for estimating property prices in Vadodara, India, using a Linear Regression model trained on real estate data. Users input details like location, BHK size, bathrooms, balcony, and sqft to get instant predictions in lakhs/cr format. Built with scikit-learn for ML, pandas .
# Vadodara House Price Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

A simple, deployable Flask web application that predicts house prices in Vadodara using a Linear Regression model trained on a dataset of 340 properties. Input details like location, BHK, bathrooms, balcony, and sqft to get estimates formatted in lakhs or crores (e.g., "₹37 lakhs").

## Features
- **ML Model**: Linear Regression with one-hot encoding for categoricals (location, h_type, size) and median imputation for missing values (e.g., yr_built).
- **User-Friendly UI**: Bootstrap form with dropdowns for location/house type/size; responsive design.
- **API Endpoint**: `/predict_api` for POST requests (JSON output for tests).
- **Retraining**: `/retrain` endpoint to regenerate model from CSV data.
- **Indian Formatting**: Outputs like "₹1.5 cr" for readability.

## Demo
- Live: [your-deployed-url] 
- Sample Prediction: 2BHK apartment in Ajwa Road (1200 sqft, 2 baths, 1 balcony) → ₹37 lakhs.

## Quick Setup
1. Clone the repo:
