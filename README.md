# Electricity Demand Forecasting — SENI

Proyek ini melakukan **peramalan beban listrik jangka pendek** (*short-term load forecasting*) dalam satuan megawatt (MW). Studi kasus menggunakan data kebutuhan listrik Republik Dominika untuk membantu perencanaan pembangkitan energi yang lebih efisien.

Selain notebook eksplorasi dan pelatihan model, repositori ini menyediakan dashboard Streamlit untuk menjalankan prediksi menggunakan model **Long Short-Term Memory (LSTM)** yang telah dilatih.

## Tujuan

- Menganalisis pola kebutuhan listrik berdasarkan waktu, cuaca, hari libur, dan akhir pekan.
- Membandingkan pendekatan baseline, *machine learning*, dan *deep learning*.
- Memprediksi kebutuhan listrik satu jam ke depan berdasarkan data 24 jam sebelumnya.
- Menyediakan antarmuka sederhana untuk mencoba model melalui unggahan CSV.

## Pendekatan

Notebook [`modeling.ipynb`](modeling.ipynb) mencakup:

1. *Exploratory Data Analysis* (EDA): distribusi data, korelasi fitur, pola harian/mingguan, serta pemeriksaan *outlier*, duplikasi, dan nilai kosong.
2. Persiapan data deret waktu: konversi `timestamp`, pengurutan waktu, pemeriksaan stasioneritas, dan pemisahan data secara kronologis (70% pelatihan, 15% validasi, 15% pengujian).
3. Perbandingan model: *naive persistence* 24 jam, Linear Regression, XGBoost, dan LSTM.
4. Evaluasi menggunakan RMSE, MAE, dan MAPE.
5. Pelatihan LSTM berbasis *sliding window* 24 jam, dengan *early stopping* dan eksperimen *hyperparameter tuning*.

Model LSTM menggunakan tujuh fitur berikut:

| Fitur | Keterangan |
| --- | --- |
| `active_power_mw` | Beban listrik aktif dalam MW |
| `apparent_temperature_c` | Suhu terasa dalam °C |
| `cloud_cover_percent` | Persentase tutupan awan |
| `is_holiday` | Penanda hari libur |
| `is_weekend` | Penanda akhir pekan |
| `hour` | Jam pengamatan |
| `month` | Bulan pengamatan |

## Struktur proyek

```text
.
├── main.py                       # Dashboard Streamlit
├── modeling.ipynb                # EDA, pelatihan, dan evaluasi model
├── requirements.txt              # Dependensi aplikasi dashboard
└── models/
    ├── model_lstm_listrik.keras  # Model LSTM terlatih
    ├── scaler_all.pkl            # Scaler seluruh fitur input
    └── scaler_target.pkl         # Scaler target beban listrik
```

## Menjalankan dashboard

Prasyarat: Python 3.10 atau versi yang kompatibel dengan TensorFlow di perangkat Anda.

```bash
git clone https://github.com/audihus/Electricity-demand-forecasting-of-the-national-interconnected-electric-system-SENI-.git
cd Electricity-demand-forecasting-of-the-national-interconnected-electric-system-SENI-
python -m venv .venv
```

Aktifkan *virtual environment*:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Instal dependensi dan jalankan aplikasi:

```bash
pip install -r requirements.txt
streamlit run main.py
```

Setelah itu, buka alamat lokal yang ditampilkan Streamlit di browser.

## Format data untuk prediksi

Dashboard membutuhkan berkas CSV dengan minimal 24 baris data historis dan kolom berikut:

```csv
timestamp,active_power_mw,apparent_temperature_c,cloud_cover_percent,is_holiday,is_weekend,hour,month
2024-01-01 00:00:00,1200.5,24.1,35,1,0,0,1
2024-01-01 01:00:00,1185.2,23.8,38,1,0,1,1
```

- `timestamp` bersifat opsional, tetapi direkomendasikan agar data diurutkan dan waktu hasil prediksi dapat ditampilkan dengan benar.
- Gunakan urutan waktu menaik.
- Sistem mengambil 24 jam terakhir sebagai input dan memprediksi beban pada jam berikutnya.
- Jika memilih indeks di dalam data, dashboard juga menampilkan nilai aktual dan selisih prediksi. Jika memilih indeks terakhir, dashboard menghasilkan prediksi satu jam ke masa depan.

Notebook dapat membuat contoh CSV melalui bagian `sample_test_data.csv`; berkas data mentah tidak disertakan dalam repositori ini.

## Menjalankan notebook

Untuk mereproduksi eksperimen, buka `modeling.ipynb` dengan Jupyter Notebook, JupyterLab, atau Google Colab. Selain dependensi di `requirements.txt`, notebook menggunakan paket berikut:

```bash
pip install seaborn statsmodels xgboost keras-tuner jupyter
```

Sesuaikan lokasi berkas dataset pada sel pembacaan CSV sebelum menjalankan notebook. Dataset perlu memuat kolom target dan fitur yang digunakan pada analisis; beberapa eksperimen pembanding juga menggunakan fitur lag pada data sumber.

## Artefak model

Folder `models/` harus tetap berada di lokasi yang sama dengan `main.py`, karena dashboard memuat tiga artefak berikut saat aplikasi dimulai:

- `model_lstm_listrik.keras`
- `scaler_all.pkl`
- `scaler_target.pkl`

## Lisensi

Belum ada lisensi yang ditentukan. Tambahkan file `LICENSE` bila proyek akan digunakan atau didistribusikan oleh pihak lain.
