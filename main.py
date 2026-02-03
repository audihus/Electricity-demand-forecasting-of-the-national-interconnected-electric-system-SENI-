import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go 
from datetime import timedelta

# --- 1. SETUP HALAMAN (Harus paling atas) ---
st.set_page_config(
    page_title="EnergyForecast AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #ff4b4b;
        padding: 10px;
        border-radius: 5px;
    }
    div.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD ASSETS (Path sudah diperbaiki) ---
@st.cache_resource
def load_assets():
    try:
        # Mengambil dari folder models/
        model = tf.keras.models.load_model('models/model_lstm_listrik.keras')
        scaler_all = joblib.load('models/scaler_all.pkl')
        scaler_target = joblib.load('models/scaler_target.pkl')
        return model, scaler_all, scaler_target
    except Exception as e:
        return None, None, None

model, scaler_all, scaler_target = load_assets()

# Cek apakah aset berhasil dimuat
if model is None:
    st.error("""
    ❌ **Error: Model tidak ditemukan!**
    
    Pastikan struktur folder kamu seperti ini:
    - app.py
    - models/
        - model_lstm_listrik.keras
        - scaler_all.pkl
        - scaler_target.pkl
    """)
    st.stop()

# --- 3. SIDEBAR (INPUT USER) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3109/3109762.png", width=50)
    st.title("⚡ EnergyForecast")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Upload Data Test (CSV)", type=['csv'])
    
    st.markdown("### ℹ️ Cara Pakai")
    st.info(
        """
        1. Upload file CSV data testing.
        2. Geser Slider untuk memilih jam prediksi.
        3. Index terakhir = Prediksi Masa Depan.
        """
    )

# --- 4. MAIN CONTENT ---
st.title("⚡ Dashboard Prediksi Beban Listrik")
st.markdown("Sistem prediksi berbasis **Long Short-Term Memory (LSTM)** dengan metode *Sliding Window*.")

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    # --- LOGIC SLIDER ---
    min_idx = 24
    max_idx = len(df) # Bisa sampai masa depan
    
    # Buat container biar rapi
    with st.container():
        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            selected_idx = st.slider("⏱️ Geser untuk memilih waktu prediksi (Index Data)", min_idx, max_idx, max_idx)
        with col_sel2:
            st.write("") # Spacer
            st.write("") 
            run_btn = st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary")

    # --- PROSES DATA ---
    cols_input = ['active_power_mw', 'apparent_temperature_c', 'cloud_cover_percent', 'is_holiday', 'is_weekend', 'hour', 'month']
    
    input_raw = df.iloc[selected_idx-24 : selected_idx][cols_input].values
    
    if len(input_raw) != 24:
        st.warning("⚠️ Data belum cukup untuk melakukan prediksi di titik ini.")
    else:
        # Tampilkan Preview Data Input (Opsional)
        with st.expander("🔍 Lihat Data Input (24 Jam Terakhir)"):
            input_df_show = df.iloc[selected_idx-24 : selected_idx].copy()
            st.dataframe(input_df_show[['timestamp', 'active_power_mw', 'apparent_temperature_c']], use_container_width=True)

        if run_btn:
            with st.spinner('Sedang memproses algoritma LSTM...'):
                # Preprocessing & Predict
                input_scaled = scaler_all.transform(input_raw)
                input_seq = input_scaled.reshape(1, 24, len(cols_input))
                
                pred_scaled = model.predict(input_seq)
                pred_mw = scaler_target.inverse_transform(pred_scaled)[0][0]
                
                # Logic Future/Past
                is_future = (selected_idx == len(df))
                
                # Setup Waktu
                if is_future:
                    last_time = df.iloc[-1]['timestamp']
                    target_time = last_time + timedelta(hours=1)
                    time_str = f"{target_time} (Masa Depan)"
                else:
                    target_time = df.iloc[selected_idx]['timestamp']
                    time_str = str(target_time)

                # --- 5. TAMPILAN HASIL (METRICS) ---
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🕒 Waktu Target", target_time.strftime("%H:%M %d-%b"), target_time.strftime("%Y"))
                
                with col2:
                    st.metric("⚡ Prediksi Beban", f"{pred_mw:.2f} MW")
                
                if not is_future:
                    actual_mw = df.iloc[selected_idx]['active_power_mw']
                    delta = pred_mw - actual_mw
                    with col3:
                        st.metric("📊 Nilai Aktual", f"{actual_mw:.2f} MW")
                    with col4:
                        # Warna error: Hijau kalau error kecil, Merah kalau besar
                        delta_color = "normal" if abs(delta) < 50 else "inverse" 
                        st.metric("📉 Selisih (Error)", f"{abs(delta):.2f} MW", delta=-delta, delta_color=delta_color)
                else:
                    with col3:
                        st.metric("📊 Nilai Aktual", "Belum Ada")
                    with col4:
                        st.info("Mode Peramal")

                # --- 6. VISUALISASI INTERAKTIF (PLOTLY) ---
                st.markdown("### 📈 Visualisasi Tren & Prediksi")
                
                # Siapkan Data Plot
                history_mw = input_raw[:, 0]
                x_history = list(range(-24, 0))
                
                fig = go.Figure()

                # 1. Garis History (Biru)
                fig.add_trace(go.Scatter(
                    x=x_history, 
                    y=history_mw,
                    mode='lines+markers',
                    name='History (24 Jam)',
                    line=dict(color='#00a8ff', width=3),
                    marker=dict(size=6)
                ))

                # 2. Titik Prediksi (Merah Kedip)
                fig.add_trace(go.Scatter(
                    x=[0], 
                    y=[pred_mw],
                    mode='markers',
                    name='Prediksi AI',
                    marker=dict(color='#ff4757', size=20, symbol='star')
                ))

                # 3. Titik Aktual (Hijau - Jika ada)
                if not is_future:
                    actual_mw = df.iloc[selected_idx]['active_power_mw']
                    fig.add_trace(go.Scatter(
                        x=[0], 
                        y=[actual_mw],
                        mode='markers',
                        name='Aktual',
                        marker=dict(color='#2ed573', size=15, symbol='circle')
                    ))
                    
                    # Garis putus-putus penghubung
                    fig.add_trace(go.Scatter(
                        x=[0, 0],
                        y=[pred_mw, actual_mw],
                        mode='lines',
                        showlegend=False,
                        line=dict(color='gray', dash='dash', width=1)
                    ))

                # Layout Cantik
                fig.update_layout(
                    title=f"Analisis Beban Listrik: {time_str}",
                    xaxis_title="Waktu (Jam ke-0 adalah Target)",
                    yaxis_title="Beban Listrik (MW)",
                    hovermode="x unified",
                    height=500,
                    template="plotly_white",
                    legend=dict(orientation="h", y=1.1)
                )

                st.plotly_chart(fig, use_container_width=True)

else:
    # Tampilan awal kalau belum upload
    st.markdown(
        """
        <div style='text-align: center; padding: 50px;'>
            <h2>👋 Selamat Datang!</h2>
            <p>Silakan upload file CSV di sidebar sebelah kiri untuk memulai simulasi.</p>
        </div>
        """, unsafe_allow_html=True
    )