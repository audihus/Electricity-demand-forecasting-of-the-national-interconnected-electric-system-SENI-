import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# --- 1. SETUP HALAMAN ---
st.set_page_config(page_title="Prediksi Listrik LSTM", layout="wide")

st.title("⚡ Dashboard Prediksi Beban Listrik (LSTM)")
st.markdown("Model Deep Learning untuk memprediksi penggunaan listrik berdasarkan data historis 24 jam terakhir.")

# --- 2. LOAD ASSETS ---
@st.cache_resource # Biar nge-loadnya sekali aja (cepet)
def load_assets():
    model = tf.keras.models.load_model('models\model_lstm_listrik.keras')
    scaler_all = joblib.load('models\scaler_all.pkl')
    scaler_target = joblib.load('models\scaler_target.pkl')
    return model, scaler_all, scaler_target

model, scaler_all, scaler_target = load_assets()

# --- 3. INPUT USER (SIDEBAR) ---
st.sidebar.header("🔧 Panel Kontrol")

# Upload File Test (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Data Test (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Pastikan pre-processing dasar (convert timestamp, dll) sesuai notebook kamu
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
    st.sidebar.success("File berhasil dimuat!")
    
    # Pilih Satu Waktu untuk Diprediksi
    # Kita butuh minimal 24 data sebelumnya
    min_idx = 24
    max_idx = len(df) - 1
    
    selected_idx = st.sidebar.slider("Pilih Index Data ke-", min_idx, max_idx, min_idx)
    
    # --- 4. PERSIAPAN DATA (SLIDING WINDOW) ---
    # Ambil 24 jam data sebelum index yang dipilih
    # Pastikan kolom urutannya SAMA PERSIS dengan saat training
    cols_input = ['active_power_mw', 'apparent_temperature_c', 'cloud_cover_percent', 'is_holiday', 'is_weekend', 'hour', 'month']
    
    # Ambil data mentah 24 jam
    input_raw = df.iloc[selected_idx-24 : selected_idx][cols_input].values
    
    # Scaling
    input_scaled = scaler_all.transform(input_raw)
    
    # Reshape ke 3D (1, 24, 7) untuk LSTM
    input_seq = input_scaled.reshape(1, 24, len(cols_input))
    
    # --- 5. PREDIKSI ---
    if st.sidebar.button("Jalankan Prediksi"):
        # Prediksi (Skala 0-1)
        pred_scaled = model.predict(input_seq)
        
        # Kembalikan ke MW Asli
        pred_mw = scaler_target.inverse_transform(pred_scaled)[0][0]
        
        # Ambil Nilai Aktual (Kunci Jawaban)
        actual_mw = df.iloc[selected_idx]['active_power_mw']
        
        # --- 6. TAMPILKAN HASIL ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Waktu Prediksi", str(df.iloc[selected_idx]['timestamp']))
        col2.metric("Prediksi Model", f"{pred_mw:.2f} MW")
        col3.metric("Nilai Aktual", f"{actual_mw:.2f} MW", delta=f"{pred_mw - actual_mw:.2f} MW")
        
        # Visualisasi Plot 24 Jam Terakhir + Prediksi
        st.subheader("Grafik 24 Jam Terakhir & Prediksi")
        
        last_24h_mw = input_raw[:, 0] # Kolom 0 adalah active_power
        
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot data 24 jam lalu
        ax.plot(range(24), last_24h_mw, label='History 24 Jam', marker='o')
        # Plot titik aktual hari ini
        ax.scatter(24, actual_mw, color='green', label='Aktual (Target)', s=100, zorder=5)
        # Plot titik prediksi
        ax.scatter(24, pred_mw, color='red', label='Prediksi LSTM', marker='x', s=100, zorder=5)
        
        ax.set_title("Input Sequence vs Prediction")
        ax.set_ylabel("Megawatt (MW)")
        ax.set_xlabel("Jam ke- (0 sampai 23 adalah history)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

else:
    st.info("Silakan upload file CSV data testing (yang sudah bersih) di sidebar sebelah kiri.")