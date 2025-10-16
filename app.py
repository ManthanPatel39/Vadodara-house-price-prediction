from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# File names
MODEL_FILE = 'vadodara_house_model.pkl'
SCALER_FILE = 'scaler.pkl'
COLUMNS_FILE = 'columns.pkl'
LOCATIONS_FILE = 'locations.pkl'

model = None
scaler = None
columns = None
locations = None

try:
    model = pickle.load(open(MODEL_FILE, 'rb'))
    scaler = pickle.load(open(SCALER_FILE, 'rb'))
    columns = pickle.load(open(COLUMNS_FILE, 'rb'))
    locations = pickle.load(open(LOCATIONS_FILE, 'rb'))
    print(f"Loaded: Model, Scaler, {len(columns)} columns, {len(locations)} locations.")
except Exception as e:
    print(f"Load error: {e} - Run /retrain to fix.")

# Hardcoded fallback if no locations.pkl
if locations is None:
    locations = [
        'Ajwa Road', 'Akota', 'Alkapuri', 'Atladra', 'Bhayli', 'Chhani', 'Fatehgunj',
        'Gorwa', 'Gotri', 'Harni', 'Karelibaug', 'Khodiyar Nagar', 'Laxmipura',
        'Madhav Pura', 'Mandvi', 'Maneja', 'Manjalpur', 'Navapura', 'New Alkapuri',
        'New Karelibaugh', 'New Sama', 'New VIP Road', 'Sama', 'Sayajipura',
        'Soma Talav', 'Vasant Vihar', 'Vasna Road', 'Vasna-Bhayli Road', 'Waghodia Road'
    ]

# Mappings from dataset uniques
htype_map = {0: 'apartment', 1: 'duplex', 2: 'pent house', 3: 'tenament', 4: 'triplex', 5: 'villa'}
size_map = {1: '1 BHK', 2: '2 BHK', 3: '3 BHK', 4: '4 BHK', 5: '5 BHK'}

# Helper for Indian price formatting
def format_price(price):
    if price >= 10000000:  # 1 cr or more
        cr = price / 10000000
        return f"₹{round(cr, 1)} cr"
    else:  # Less than 1 cr, in lakhs
        lakhs = price / 100000
        return f"₹{round(lakhs, 0)} lakhs"

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/retrain', methods=['GET'])
def retrain():
    """Retrains and saves artifacts (visit in browser once)."""
    try:
        df = pd.read_csv('vadodara_house_price_dataset_new.csv')
        df['yr_built'].fillna(df['yr_built'].median(), inplace=True)  # Impute missing
        
        # Update mappings from data
        global locations, htype_map, size_map
        locations = sorted(df['location'].dropna().unique().tolist())
        htypes = sorted(df['h_type'].dropna().unique().tolist())
        sizes = sorted(df['size'].dropna().unique().tolist())
        htype_map = {i: htypes[i] for i in range(len(htypes))}
        size_map = {i: sizes[i] for i in range(len(sizes))}
        
        # Train and save
        X = pd.get_dummies(df.drop('price', axis=1), drop_first=True)
        y = df['price']
        X_scaled = StandardScaler().fit_transform(X)
        X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        new_model = LinearRegression().fit(X_train, y_train)
        
        pickle.dump(new_model, open(MODEL_FILE, 'wb'))
        pickle.dump(StandardScaler().fit(X), open(SCALER_FILE, 'wb'))
        pickle.dump(list(X.columns), open(COLUMNS_FILE, 'wb'))
        pickle.dump(locations, open(LOCATIONS_FILE, 'wb'))
        
        print(f"Retrained: {len(X.columns)} features.")
        return "Retrained! Restart app."
    except Exception as e:
        return f"Retrain error: {e}"

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get form data (safe defaults)
        h_type_idx = int(request.form.get('h_type', 0))
        location = request.form.get('location', locations[0])  # First as baseline
        size_idx = int(request.form.get('size', 2))
        bath = float(request.form.get('bath', 2))
        balcony = float(request.form.get('balcony', 1))
        total_sqft = float(request.form.get('total_sqft', 1200))

        # Map to strings for one-hot
        h_type = htype_map.get(h_type_idx, 'apartment')
        size = size_map.get(size_idx, '2 BHK')

        # Create FULL input DataFrame (270 columns, zeros baseline)
        input_df = pd.DataFrame(0.0, index=[0], columns=columns)
        
        # Set user numerics
        input_df.at[0, 'total_sqft'] = total_sqft
        input_df.at[0, 'bathroom'] = bath
        input_df.at[0, 'balcony'] = balcony
        
        # One-hot location
        loc_col = f'location_{location}'
        if loc_col in input_df.columns:
            input_df.at[0, loc_col] = 1
        
        # One-hot h_type
        htype_col = f'h_type_{h_type}'
        if htype_col in input_df.columns:
            input_df.at[0, htype_col] = 1
        
        # One-hot size
        size_col = f'size_{size}'
        if size_col in input_df.columns:
            input_df.at[0, size_col] = 1
        
        # Defaults for other features
        defaults = {
            'yr_built': 2015.0,
            'furniture': 0.0,
            'amenities': 1.0,
            'market': 1.0, 'office': 1.0, 'school': 1.0, 'college': 0.0,
            'hospital': 1.0, 'population': 1.0, 'railway': 0.0, 'airport': 0.0,
            'on_road': 1.0, 'air_quality': 1.0, 'restaurant': 1.0, 'park': 1.0
        }
        for feat, val in defaults.items():
            if feat in input_df.columns:
                input_df.at[0, feat] = val
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        # Format price
        formatted_price = format_price(prediction)
        
        # Detect browser form vs. API test
        is_browser = 'text/html' in request.headers.get('Accept', '')
        if is_browser:
            return render_template('index.html', prediction_text=f'Estimated House Price: {formatted_price}')
        else:
            return jsonify({'prediction': formatted_price})

    except ValueError as e:
        return jsonify({'error': f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)