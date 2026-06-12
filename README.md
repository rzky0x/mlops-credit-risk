# Submission 1: Credit Risk Prediction ML Pipeline

Nama: rzky0x

Username dicoding: rzky0x

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) |
| Masalah | Memprediksi risiko kredit (loan default) berdasarkan data profil peminjam. Masalah ini penting bagi lembaga keuangan untuk mengidentifikasi peminjam berisiko tinggi sebelum memberikan pinjaman, sehingga dapat meminimalkan risiko kerugian finansial. |
| Solusi machine learning | Membangun model binary classification menggunakan Deep Neural Network (DNN) untuk memprediksi apakah seorang peminjam akan mengalami default (gagal bayar) atau tidak. Target performa: AUC ≥ 0.7 dan Binary Accuracy ≥ 0.7. |
| Metode pengolahan | **Fitur Numerik** (person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length): Dinormalisasi menggunakan z-score scaling (`tft.scale_to_z_score`). **Fitur Kategorikal** (person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file): Di-encode menggunakan vocabulary lookup (`tft.compute_and_apply_vocabulary`). Missing values diisi dengan 0 (numerik) atau string kosong (kategorikal). |
| Arsitektur model | DNN Classifier dengan arsitektur: Input Layer → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) → Dense(64, ReLU) → BatchNorm → Dropout(0.3) → Dense(1, Sigmoid). Optimizer: Adam (lr=0.001). Loss: Binary Crossentropy. |
| Metrik evaluasi | **AUC (Area Under ROC Curve)**: Mengukur kemampuan model membedakan antara kelas positif dan negatif. **Binary Accuracy**: Mengukur proporsi prediksi yang benar. Threshold blessing: AUC ≥ 0.7, Binary Accuracy ≥ 0.7. |
| Performa model | Model berhasil mencapai performa yang memenuhi threshold yang ditetapkan dan mendapatkan status "blessed" dari komponen Evaluator. Detail metrik dapat dilihat pada output komponen Evaluator di notebook `rzky0x-pipeline.ipynb`. |
| Opsi deployment | Model di-deploy menggunakan **TensorFlow Serving** dalam **Docker container** di platform **Railway**. Railway dipilih karena mendukung deployment container Docker secara gratis dan menyediakan public URL otomatis. |
| Web app | [credit-risk-model](https://<YOUR_RAILWAY_URL>.up.railway.app/v1/models/credit-risk-model/metadata) |
| Monitoring | Model serving dimonitor menggunakan **Prometheus** yang mengumpulkan metrics dari endpoint `/monitoring/prometheus/metrics` pada TF Serving. Dashboard monitoring dibuat menggunakan **Grafana** untuk visualisasi metrics seperti request count, latency, dan model status. Prometheus dikonfigurasi dengan scrape interval 5 detik untuk memastikan monitoring real-time. |

## Struktur Proyek

```
.
├── data/                              # Dataset CSV
│   └── credit_risk_dataset.csv
├── modules/                           # Modul Python untuk pipeline
│   ├── __init__.py
│   ├── transform_module.py            # Modul preprocessing/transform
│   └── trainer_module.py              # Modul training model
├── rzky0x-pipeline/                   # Pipeline artifacts
│   ├── metadata/                      # ML Metadata
│   └── serving_model/                 # Model serving
├── monitoring/                        # Setup monitoring
│   ├── Dockerfile                     # Dockerfile Prometheus
│   ├── prometheus.config              # Config monitoring TF Serving
│   └── prometheus.yml                 # Konfigurasi scrape Prometheus
├── Dockerfile                         # Dockerfile deployment
├── requirements.txt                   # Dependencies
├── rzky0x-pipeline.ipynb              # Notebook pipeline utama
├── rzky0x-testing.ipynb               # Notebook testing
└── README.md                         # Dokumentasi (file ini)
```

## Cara Menjalankan

### 1. Setup Environment
```bash
python -m venv mlops-env
source mlops-env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Jalankan Pipeline
Buka dan jalankan notebook `rzky0x-pipeline.ipynb` di Jupyter.

### 3. Build & Deploy Docker
```bash
# Build image
docker build -t credit-risk-model .

# Test locally
docker run -p 8501:8501 credit-risk-model

# Test endpoint
curl http://localhost:8501/v1/models/credit-risk-model/metadata
```

### 4. Deploy ke Railway
1. Push project ke GitHub repository
2. Hubungkan repository ke Railway
3. Railway akan otomatis detect Dockerfile dan melakukan build
4. Setelah deploy berhasil, dapatkan public URL

### 5. Monitoring dengan Prometheus
```bash
cd monitoring
docker build -t prometheus-monitoring .
docker run -p 9090:9090 prometheus-monitoring
```
Akses Prometheus dashboard di `http://localhost:9090`

### 6. Grafana Dashboard
```bash
docker run -d -p 3000:3000 grafana/grafana:latest
```
1. Akses Grafana di `http://localhost:3000` (login: admin/admin)
2. Tambahkan data source Prometheus (`http://localhost:9090`)
3. Buat dashboard dengan query: `tensorflow:serving:request_count`
