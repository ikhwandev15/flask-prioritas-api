from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model dan encoder
model = joblib.load('model.pkl')
le_kategori = joblib.load('le_kategori.pkl')
le_prioritas = joblib.load('le_prioritas.pkl')

@app.route('/')
def index():
    return "✅ Flask + Random Forest siap digunakan!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        kategori = data['kategori']
        selisih = float(data['selisih_menit'])
        is_weekend = int(data.get('is_weekend', 0))

        kategori_encoded = le_kategori.transform([kategori])[0]
        input_df = pd.DataFrame([[kategori_encoded, selisih, is_weekend]],
                                columns=['kategori_encoded', 'selisih_menit', 'is_weekend'])

        hasil = model.predict(input_df)
        prioritas = le_prioritas.inverse_transform(hasil)

        return jsonify({'prioritas': prioritas[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
