import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Data penggunaan listrik
data = {
    'rata_rata_pemakaian': [10, 15, 200, 8, 300, 12, 9, 250],
    'jumlah_hari_penggunaan': [30, 29, 5, 30, 2, 28, 30, 6],
    'variansi_pemakaian': [1, 2, 10, 0.5, 15, 1.5, 0.8, 12]
}

# Membuat DataFrame
df = pd.DataFrame(data)

print("Data Penggunaan Listrik:")
print(df)

# Memilih fitur untuk model
X = df[['rata_rata_pemakaian', 'jumlah_hari_penggunaan', 'variansi_pemakaian']]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Isolation Forest
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X_scaled)

# Prediksi anomali
predictions = model.predict(X_scaled)

# Menambahkan hasil prediksi ke DataFrame
df['prediksi'] = predictions
df['status'] = df['prediksi'].apply(lambda x: 'Normal' if x == 1 else 'Anomali')

print("\nHasil Deteksi Anomali:")
print(df)

# Fungsi untuk deteksi kasus baru
def deteksi_anomali(rata_rata_pemakaian, jumlah_hari_penggunaan, variansi_pemakaian):
    input_data = np.array([[rata_rata_pemakaian, jumlah_hari_penggunaan, variansi_pemakaian]])
    input_scaled = scaler.transform(input_data)
    
    prediksi = model.predict(input_scaled)
    
    if prediksi == 1:
        return "Penggunaan Normal"
    else:
        return "Penggunaan Mencurigakan (Anomali)"

# Input data baru untuk deteksi
print("\nMasukkan data penggunaan listrik:")
rata_rata_pemakaian = float(input("Rata-rata Pemakaian (kWh): "))
jumlah_hari_penggunaan = int(input("Jumlah Hari Penggunaan: "))
variansi_pemakaian = float(input("Variansi Pemakaian: "))

hasil = deteksi_anomali(rata_rata_pemakaian, jumlah_hari_penggunaan, variansi_pemakaian)
print(f"Hasil Prediksi: {hasil}")