from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
le_kategori = joblib.load("le_kategori.pkl")
le_prioritas = joblib.load("le_prioritas.pkl")

@app.route('/')
def home():
    return "✅ Flask + Random Forest siap digunakan!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    kategori = data.get("kategori")

    if not kategori:
        return jsonify({"error": "Field 'kategori' harus ada"}), 400

    kategori_encoded = le_kategori.transform([kategori])
    prediction_encoded = model.predict([[kategori_encoded[0]]])
    prediction = le_prioritas.inverse_transform(prediction_encoded)

    return jsonify({"kategori": kategori, "prioritas": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
