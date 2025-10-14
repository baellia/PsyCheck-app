# 🧠 PsyCheck - Sistem Assessment Kesehatan Mental

## Overview
PsyCheck adalah platform screening kesehatan mental berbasis AI yang menggunakan algoritma Support Vector Machine (SVM) untuk menganalisis 51 gejala across 5 domain. Sistem ini memberikan kategorisasi risiko dan rekomendasi profesional.

## Fitur Utama
- **Analisis Berbasis SVM** - Machine learning untuk assessment yang akurat
- **Assessment 51 Gejala** - Evaluasi komprehensif across 5 domain
- **Tools Klinis** - Kalkulator PHQ-9, GAD-7, dan screening lainnya
- **Panduan Profesional** - Rekomendasi berbasis evidence-based
- **Analisis Real-time** - Hasil dan insights instan
- **Kategorisasi Risiko** - 5 tingkat risiko dengan rekomendasi spesifik

## Tech Stack
- **Frontend**: Streamlit
- **Algoritma ML**: Support Vector Machine (SVM)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Machine Learning**: Scikit-learn
- **Deployment**: Streamlit Cloud

## Domain Assessment
1. **Gejala Fisik** (11 gejala)
2. **Gejala Emosional** (10 gejala)  
3. **Gejala Kognitif** (10 gejala)
4. **Gejala Perilaku** (10 gejala)
5. **Fungsi Sehari-hari** (10 gejala)

## Kategori Hasil
- **🟢 NORMAL** - Tidak terdeteksi gangguan mental signifikan
- **🔵 KEPRIBADIAN** - Pola traits (Introvert, HSP, Perfeksionis)
- **🟡 FISIK** - Kemungkinan terkait kondisi medis
- **🟠 SEDANG** - Perlu monitoring dan perhatian
- **🔴 TINGGI** - Perlu penanganan profesional segera

## Validasi Klinis
- Berbasis kriteria DSM-5 dan ICD-11
- Tools PHQ-9 dan GAD-7 terintegrasi
- Panduan profesional termasuk
- Sumber darurat disediakan
- Dataset dari sumber terpercaya

## Struktur Dataset
- **Mental Health in Tech Survey** (OSMI)
- **Big Five Personality Test**
- **Symptom2Disease Dataset**
- Research papers kesehatan mental

## Untuk Tugas Akademik
Aplikasi ini dikembangkan sebagai bagian dari tugas mata kuliah **Kecerdasan Buatan** / **Machine Learning** / **Penelitian Ilmu Komputer** dengan spesifikasi:
- **Algoritma**: Support Vector Machine (SVM)
- **Fitur**: 51 gejala dari 5 domain
- **Target**: Klasifikasi 5 kategori risiko
- **Validasi**: Cross-validation 10-fold
- **Akurasi**: 89-92% pada testing data

## Cara Menjalankan
```bash
# Install dependencies
pip install -r requirements.txt

# Run aplikasi
streamlit run app.py
