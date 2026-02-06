# Klasifikasi Peraturan OJK

Sistem klasifikasi otomatis untuk mengelompokkan Peraturan Otoritas Jasa Keuangan (POJK) ke dalam departemen pengawasan yang sesuai. Dibangun menggunakan pendekatan Natural Language Processing dengan pipeline TF-IDF dan SGD Classifier.

Demo: [klasifikasi-peraturan-ojk.vercel.app](https://klasifikasi-peraturan-ojk.vercel.app)

---

## Latar Belakang

OJK memiliki empat departemen pengawasan utama yang masing-masing menangani bidang regulasi berbeda:

| Departemen | Cakupan |
|---|---|
| **Perbankan** | Bank umum, BPR, bank syariah, unit usaha syariah |
| **Pasar Modal** | Saham, obligasi, reksa dana, perusahaan efek, emiten |
| **IKNB** | Asuransi, dana pensiun, lembaga pembiayaan, penjaminan |
| **ITSK** | Fintech, inovasi teknologi keuangan, layanan digital |

Pengelompokan peraturan secara manual memakan waktu. Project ini mengotomasi proses tersebut menggunakan machine learning berbasis teks regulasi.

## Arsitektur

```
docs_POJK/                  # Koleksi file PDF peraturan OJK
    |
    v
extract_pdf.py              # Ekstraksi teks dari PDF (pdfplumber)
    |
    v
output_pojk.csv             # Teks mentah hasil ekstraksi
    |
    v
classify_department.py      # Pelabelan awal berbasis keyword matching
    |
    v
output_pojk_classified.csv  # Dataset berlabel (filename, content, department)
    |
    v
train_model.py              # Training model TF-IDF + SGD Classifier
    |
    v
model_klasifikasi_ojk.joblib  # Model tersimpan
    |
    v
app.py / api/index.py       # Web app Flask (lokal / Vercel)
```

## Struktur File

```
.
├── api/
│   └── index.py                    # Entry point serverless (Vercel)
├── templates/
│   └── index.html                  # Antarmuka web (Bootstrap 5)
├── docs_POJK/                      # Direktori PDF peraturan OJK
├── app.py                          # Flask app (development lokal)
├── extract_pdf.py                  # Ekstraksi teks dari PDF
├── classify_department.py          # Pelabelan berbasis keyword
├── train_model.py                  # Training dan evaluasi model
├── model_klasifikasi_ojk.joblib    # Model hasil training
├── notebook_klasifikasi_ojk.ipynb  # Notebook eksplorasi dan tuning
├── requirements.txt                # Dependensi Python
├── vercel.json                     # Konfigurasi deployment Vercel
└── README.md
```

## Metode

### Preprocessing

Teks dari setiap dokumen PDF diekstrak menggunakan pdfplumber, lalu dibersihkan dengan menghapus angka, karakter khusus, dan whitespace berlebih. Hasilnya disimpan dalam format CSV.

### Pelabelan

Pelabelan awal dilakukan secara rule-based menggunakan keyword matching. Setiap departemen memiliki daftar kata kunci. Dokumen diklasifikasikan berdasarkan frekuensi kemunculan kata kunci dari masing-masing departemen.

### Model

Beberapa model dibandingkan melalui cross-validation:

| Model | Akurasi CV (5-fold) |
|---|---|
| Multinomial Naive Bayes | 81.0% |
| Logistic Regression | 83.3% |
| Linear SVC | 85.0% |
| Random Forest | 78.3% |
| **SGD Classifier** | **88.6%** |

Model terbaik menggunakan konfigurasi:
- **Vectorizer**: TfidfVectorizer, max_features=10000, ngram_range=(1,3), sublinear_tf=True
- **Classifier**: SGDClassifier, loss=modified_huber, alpha=0.0001, max_iter=5000
- Hyperparameter dituning menggunakan GridSearchCV dengan StratifiedKFold

### Dataset

- 43 dokumen POJK dari tahun 2024-2025
- Distribusi label: Perbankan (15), Pasar Modal (15), IKNB (8), ITSK (5)

## Cara Penggunaan

### Prasyarat

- Python 3.10 atau lebih baru
- pip

### Instalasi

```bash
git clone https://github.com/kamilramadhan/Klasifikasi-Peraturan-OJK.git
cd Klasifikasi-Peraturan-OJK
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pdfplumber
```

### Menjalankan Pipeline dari Awal

1. Letakkan file PDF peraturan OJK di dalam direktori `docs_POJK/`.

2. Ekstrak teks dari PDF:
   ```bash
   python extract_pdf.py
   ```
   Menghasilkan `output_pojk.csv`.

3. Beri label berdasarkan keyword:
   ```bash
   python classify_department.py
   ```
   Menghasilkan `output_pojk_classified.csv`.

4. Training model:
   ```bash
   python train_model.py
   ```
   Menghasilkan `model_klasifikasi_ojk.joblib`.

### Menjalankan Web App (Lokal)

```bash
python app.py
```

Buka `http://127.0.0.1:5000` di browser. Ketik teks pengaduan atau deskripsi peraturan, lalu tekan tombol prediksi untuk melihat departemen yang sesuai beserta tingkat confidence.

### Deployment (Vercel)

Project ini sudah dikonfigurasi untuk Vercel. File `api/index.py` berfungsi sebagai serverless function dan `vercel.json` mengatur routing. Untuk deploy ulang:

```bash
vercel --prod
```

## Dependensi

| Package | Versi | Fungsi |
|---|---|---|
| flask | 3.1.1 | Web framework |
| scikit-learn | 1.8.0 | TF-IDF, SGD Classifier, evaluasi model |
| joblib | 1.5.1 | Serialisasi model |
| pdfplumber | - | Ekstraksi teks dari PDF |
| pandas | - | Manipulasi data |

## Lisensi

MIT
