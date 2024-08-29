# Aplikasi Prediksi Diabetes

Aplikasi web untuk prediksi diabetes menggunakan machine learning. Proyek ini dibangun menggunakan Flask dan menyediakan antarmuka sederhana bagi pengguna untuk memasukkan data kesehatan mereka dan mendapatkan prediksi kemungkinan terkena diabetes.

## Fitur

- **Model Machine Learning**: Aplikasi ini menggunakan model yang telah dilatih (`diabetes_model.pkl`) untuk memprediksi diabetes.
- **Antarmuka Pengguna**: Pengguna dapat memasukkan data mereka melalui sebuah formulir dan menerima prediksi.
- **Visualisasi Hasil**: Hasil prediksi ditampilkan dalam format yang mudah dipahami.

## Struktur Proyek

- `app.py`: File aplikasi utama yang berisi route Flask.
- `static/`: Berisi file statis seperti stylesheet CSS.
- `templates/`: Berisi template HTML (`index.html`, `form.html`, `result.html`).
- `model.py`: Script yang digunakan untuk melatih model machine learning.
- `requirements.txt`: Berisi daftar dependensi Python yang diperlukan untuk proyek ini.
- `data_diabetes_puskesmas_tegalbuleud.xlsx`: Dataset yang digunakan untuk melatih model.
- `diabetes_model.pkl`: Model machine learning yang telah dilatih.
- `scaler.pkl`: Scaler yang digunakan untuk normalisasi data input.
