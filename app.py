import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # لتفعيل CORS لجميع المسارات



# Load model, scaler, and outlier handler
with open("models/Heat.Stroke_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/Scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/outlier_handler.pkl", "rb") as f:
    outlier_handler = pickle.load(f)

@app.route('/')
def home():
    return "🔥 Heat Stroke Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required = ['Environmental temperature (C)', 'Heart / Pulse rate (b/min)', 'Sweating', 'Patient temperature']
        if not data or not all(col in data for col in required):
            return jsonify({"error": "Missing or invalid input fields", "required": required}), 400

        input_df = pd.DataFrame([data])
        
        # تصحيح استخدام معالج القيم الشاذة
        input_df[['Environmental temperature (C)', 'Heart / Pulse rate (b/min)']] = \
            outlier_handler.transform(input_df[['Environmental temperature (C)', 'Heart / Pulse rate (b/min)']])
        
        # توسيع البيانات
        scaled = scaler.transform(input_df)
        
        # التنبؤ
        pred = model.predict(scaled)[0]
        
        return jsonify({"prediction": "Heat Stroke" if pred == 1 else "No Heat Stroke"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()