import requests
import json

url = 'http://127.0.0.1:5000/predict_api'
data = {
    'h_type': '0',  # Apartment
    'location': 'Ajwa Road',
    'size': '2',
    'bath': '2',
    'balcony': '1',
    'total_sqft': '1200'
}

try:
    response = requests.post(url, data=data)
    print(f"Raw response: {response.text}")  # Debug: shows JSON
    if response.status_code == 200:
        result = response.json()
        pred = result.get('prediction')
        if pred:
            print(f"✅ Predicted House Price: {pred}")
        else:
            print("❌ No 'prediction' key in response")
            print("Full result:", result)
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"❌ Connection error: {e}")
except json.JSONDecodeError as e:
    print(f"❌ JSON parse error: {e}")