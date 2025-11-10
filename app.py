#Importing required libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

#App Initialisation
app = Flask(__name__)
CORS(app)  # Allow all origins for local development

# Load model, scaler and polynomial transformer
model = joblib.load("solar_output_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly_transformer.pkl")

# Define feature order used in training
FEATURE_NAMES = [
    "location_type",
    "avg_temperature_C",
    "cloud_coverage_percent",
    "wind_speed_kmph",
    "humidity_percent",
    "panel_age_years",
    "panel_efficiency_percent",
    "solar_radiation_kWh_m2"
]

@app.route('/ping')
def ping():
    return jsonify({"status": "ok"}), 200
    
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure all expected features are present
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {', '.join(missing)}"}), 400

        # Extract, scale and transform input values
        input_array = np.array([data[f] for f in FEATURE_NAMES]).reshape(1, -1)
        X = pd.DataFrame(input_array, columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X)
        X_poly = poly.transform(X_scaled)

        
        # Predict and ensure non-negative
        prediction = max(0, model.predict(X_poly)[0])

        return jsonify({
            "normalized_output_kWh": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)




