import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('dataset_tugas_prioritas.csv', encoding='ISO-8859-1')

le_kategori = LabelEncoder()
le_prioritas = LabelEncoder()
df['kategori_encoded'] = le_kategori.fit_transform(df['kategori'])
df['prioritas_encoded'] = le_prioritas.fit_transform(df['prioritas'])

df['deadline'] = pd.to_datetime(df['deadline'])
df['waktu_dibuat'] = pd.to_datetime(df['waktu_dibuat'])
df['selisih_menit'] = (df['deadline'] - df['waktu_dibuat']).dt.total_seconds() / 60
df['is_weekend'] = (df['deadline'].dt.dayofweek >= 5).astype(int)

X = df[['kategori_encoded', 'selisih_menit', 'is_weekend']]
y = df['prioritas_encoded']
model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
joblib.dump(le_kategori, 'le_kategori.pkl')
joblib.dump(le_prioritas, 'le_prioritas.pkl')

print("✅ Model berhasil disimpan sebagai model.pkl")