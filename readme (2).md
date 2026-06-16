# 🏠 Yogyakarta House Price Prediction

> **Multi-model machine learning pipeline** untuk prediksi harga rumah di Yogyakarta menggunakan 6 algoritma regresi, analisis geospasial, dan teknik anti-overfitting.

---

## 📊 Model Performance Summary

| Model | MAE | RMSE | R² Test | MAPE | CV R² | Gap | Status |
|-------|-----|------|---------|------|-------|-----|--------|
| 🥇 Random Forest | Rp 285 Jt | Rp 515 Jt | **0.7714** | 19.32% | 0.8060 ± 0.0242 | 0.2024 | ⚠️ Overfit |
| 🥈 Stacking Ensemble | Rp 285 Jt | Rp 517 Jt | 0.7694 | 19.16% | N/A | 0.2047 | ⚠️ Overfit |
| 🥉 Gradient Boosting | Rp 292 Jt | Rp 524 Jt | 0.7631 | 20.40% | 0.7727 ± 0.0294 | 0.2295 | ⚠️ Overfit |
| XGBoost | Rp 296 Jt | Rp 548 Jt | 0.7405 | 20.04% | 0.7824 ± 0.0217 | 0.2582 | ⚠️ Overfit |
| LightGBM | Rp 309 Jt | Rp 532 Jt | 0.7551 | 21.28% | 0.7667 ± 0.0243 | 0.2184 | ⚠️ Overfit |
| Ridge (Poly) | Rp 463 Jt | Rp 656 Jt | 0.6280 | 39.03% | N/A | 0.0204 | ✅ Stabil |
| ElasticNet (Poly) | Rp 644 Jt | Rp 836 Jt | 0.3965 | 63.10% | N/A | 0.0181 | ✅ Stabil |

> **Best model → Random Forest** (R² Test: 0.7714, MAE: Rp 285 Jt)
> **Prediksi sampel** (Depok, Sleman | 2KT 1KM 36m²): Ensemble Top-3 = **Rp 400.890.376**

---

## 🚩 Problem Statement

Pasar properti Yogyakarta berkembang pesat seiring pertumbuhan sektor pendidikan, pariwisata, dan hunian. Namun, **tidak ada standar harga yang transparan dan terukur**, sehingga:

- 🔴 Pembeli kesulitan menentukan apakah harga listing wajar atau kemahalan
- 🔴 Penjual/developer sulit menetapkan harga kompetitif berbasis data
- 🔴 Investor tidak memiliki referensi harga berbasis lokasi dan spesifikasi properti

---

## 💡 Solution

Membangun sistem prediksi harga rumah berbasis **machine learning multi-model** yang:
''
Input: Lokasi + Luas Tanah + Luas Bangunan + Kamar Tidur + Kamar Mandi + Carport

↓

Feature Engineering (area_ratio, room_density, log-transform)

↓

6 Model Regresi + Hyperparameter Tuning (GridSearchCV / BayesSearchCV)

↓

Stacking Ensemble (RF + GB + XGB → Ridge meta-learner)

↓

Output: Prediksi Harga + Interval Kepercayaan + Analisis Geospasial
''
---

## 🔬 Apa yang Dilakukan dalam Model

### 1️⃣ Data & Preprocessing
- **Dataset:** 2.019 listing rumah Yogyakarta (setelah filter: 1.822 baris)
- **Outlier removal:** IQR 5–95 percentile pada kolom `price`
- **Imputation:** Median untuk numerik, Most Frequent untuk kategorik
- **Encoding:** OneHotEncoder dengan `handle_unknown='ignore'`
- **Scaling:** StandardScaler pada semua fitur numerik

### 2️⃣ Feature Engineering

| Fitur Baru | Formula | Tujuan |
|------------|---------|--------|
| `area_ratio` | `building_area / (surface_area + 1)` | Kepadatan bangunan |
| `room_density` | `(bed + bath) / (building_area + 1)` | Kepadatan ruangan per m² |
| `total_rooms` | `bed + bath + carport` | Total fasilitas |
| `log_surface` | `log1p(surface_area)` | Normalisasi skew distribusi |
| `log_building` | `log1p(building_area)` | Normalisasi skew distribusi |

### 3️⃣ Model & Optimasi
''
Random Forest     → GridSearchCV (48 kombinasi × 5 fold = 240 fits)

Gradient Boosting → GridSearchCV (32 kombinasi × 5 fold = 160 fits)

XGBoost           → GridSearchCV (96 kombinasi × 5 fold = 480 fits)

LightGBM          → GridSearchCV (64 kombinasi × 5 fold = 320 fits)

Ridge             → GridSearchCV + PolynomialFeatures (degree=2)

ElasticNet        → GridSearchCV + PolynomialFeatures (degree=2)

Stacking          → RF + GB + XGB → Ridge meta-learner (cv=5)
''
### 4️⃣ Evaluasi Multi-Metrik

| Metrik | Deskripsi | Kenapa Penting |
|--------|-----------|----------------|
| **MAE** | Mean Absolute Error | Error rata-rata dalam satuan Rupiah |
| **MSE** | Mean Squared Error | Penalti lebih besar untuk error besar |
| **RMSE** | Root MSE | Interpretasi sama dengan MAE tapi sensitif outlier |
| **R²** | Koefisien Determinasi | Proporsi variansi yang dijelaskan model |
| **MAPE** | Mean Absolute Percentage Error | Error relatif, bebas skala harga |
| **CV R²** | Cross-Validation R² ± Std | Stabilitas model pada data berbeda |
| **Gap Train-Test** | R²Train − R²Test | Indikator overfitting |

### 5️⃣ Best Hyperparameter (Random Forest)

```python
{
  'bootstrap': True,
  'max_depth': None,
  'max_features': 'sqrt',
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 500
}
```

### 6️⃣ Catatan Overfitting

> ⚠️ Semua tree-based model menunjukkan gap train-test **0.20–0.26** yang mengindikasikan overfitting cukup signifikan. Ini kemungkinan disebabkan oleh:
> - **69 lokasi unik** yang di-encode menjadi banyak fitur sparse (OHE)
> - Ukuran dataset yang relatif kecil (1.822 baris) vs kompleksitas ruang fitur
> - Data properti memiliki varians harga tinggi antar mikro-lokasi

**Rekomendasi perbaikan lanjutan:**
- Target Encoding atau Embedding untuk fitur lokasi (ganti OHE)
- Tambah data (scraping lebih banyak listing)
- Regularisasi lebih ketat: turunkan `max_depth`, naikkan `min_samples_leaf`
- Pertimbangkan `log(price)` sebagai target untuk menstabilkan varians

---

## 🗺️ Analisis Geospasial

- Peta interaktif (Folium + Google Maps tiles) dengan **HeatMap** harga dan **MarkerCluster** per kecamatan
- Setiap marker menampilkan: median harga, rata-rata harga, jumlah listing
- Output disimpan sebagai HTML interaktif di `backend/static/maps/yogyakarta_geo.html`

**Hotspot harga tertinggi:**
''
🔵 Ngaglik, Sleman     → 303 listing (terbanyak)

🔵 Depok, Sleman       → 197 listing

🔵 Kalasan, Sleman     → 144 listing

🔵 Mlati, Sleman       → 122 listing

🔵 Banguntapan, Bantul → 117 listing
''
---

## 🛠️ Tech Stack

### Language & Environment
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

### Machine Learning
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02457A?style=flat&logo=lightgbm&logoColor=white)

### Data & Visualization
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![Folium](https://img.shields.io/badge/Folium_(GIS)-77B829?style=flat&logo=folium&logoColor=white)

### Model Persistence
![joblib](https://img.shields.io/badge/joblib-Model_Saving-lightgrey?style=flat)

---

## 📁 Struktur Proyek
''
📦 house-price-prediction/

├── 📂 dataset/

│   └── 📂 preprocessing/

│       └── yogyakarta_clean.csv

├── 📂 notebook/

│   └── house_price_prediction_benchmark.ipynb

├── 📂 house-price-prediction-website/

│   └── 📂 backend/

│       ├── 📂 models/

│       │   ├── random_forest.pkl

│       │   ├── gradient_boosting.pkl

│       │   ├── xgboost.pkl

│       │   ├── lightgbm.pkl

│       │   ├── ridge_poly.pkl

│       │   ├── elasticnet_poly.pkl

│       │   ├── yogyakarta_best_model.pkl

│       │   └── benchmark_summary.pkl

│       └── 📂 static/maps/

│           └── yogyakarta_geo.html

└── README.md
''
---

## ⚙️ Cara Menjalankan

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/username/yogyakarta-house-price-prediction.git
cd yogyakarta-house-price-prediction

pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn folium joblib scipy
# Opsional (Bayesian optimization):
pip install scikit-optimize
```

### 2. Jalankan Notebook
```bash
jupyter notebook notebook/house_price_prediction_benchmark.ipynb
```

### 3. Load Model untuk Prediksi
```python
import joblib, pandas as pd, numpy as np

model = joblib.load('backend/models/yogyakarta_best_model.pkl')

new_data = pd.DataFrame({
    'location':      ['Depok, Sleman'],
    'bed':           [2],
    'bath':          [1],
    'carport':       [0],
    'surface_area':  [40],
    'building_area': [36],
    'area_ratio':    [36/41],
    'room_density':  [3/37],
    'total_rooms':   [3],
    'log_surface':   [np.log1p(40)],
    'log_building':  [np.log1p(36)],
})

pred = model.predict(new_data)[0]
print(f"Prediksi Harga: Rp {pred:,.0f}")
# Output: Prediksi Harga: Rp 457,431,778
```

---

## 🧠 Skills yang Digunakan
''
Machine Learning       → Supervised Learning, Ensemble Methods, Stacking

Feature Engineering    → Log-transform, Interaction Features, Ratio Features

Hyperparameter Tuning  → GridSearchCV, BayesSearchCV, Cross-Validation

Model Evaluation       → MAE, RMSE, R², MAPE, CV Score, Overfitting Analysis

Data Preprocessing     → Imputation, Scaling, OneHotEncoding, Outlier Removal

Geospatial Analysis    → Folium, HeatMap, MarkerCluster, GIS Visualization

Data Visualization     → Matplotlib, Seaborn, Learning Curve, Residual Analysis

MLOps (Basic)          → Model Persistence (joblib), Pipeline Architecture

Python                 → NumPy, Pandas, scikit-learn, XGBoost, LightGBM
''
---

## 👤 Author

**Puteri Amelia Azli**
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/puteriazli)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/puteriazli)

---

<p align="center">
  <sub>Built with ❤️ for Yogyakarta property market transparency</sub>
</p>
