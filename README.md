# Klasifikasi Peraturan OJK

Sistem klasifikasi otomatis untuk mengelompokkan Peraturan Otoritas Jasa Keuangan (POJK) ke dalam departemen pengawasan yang sesuai berdasarkan isi dokumen. Dibangun menggunakan pendekatan NLP dengan pipeline TF-IDF dan Multinomial Naive Bayes.

Demo: [klasifikasi-peraturan-ojk.vercel.app](https://klasifikasi-peraturan-ojk.vercel.app)

---

## Latar Belakang

Berdasarkan Struktur Organisasi OJK-Wide (Eksisting), terdapat enam bidang pengawasan utama di bawah Dewan Komisioner:

| Kode Bidang | Departemen | Cakupan |
|---|---|---|
| Bidang 3 | **Perbankan** | Bank umum, BPR, BPRS, bank syariah, unit usaha syariah |
| Bidang 4 | **Pasar Modal** | Efek, reksa dana, manajer investasi, emiten, keuangan derivatif, bursa karbon |
| Bidang 5 | **Perasuransian** | Perusahaan asuransi, reasuransi, penjaminan, dana pensiun |
| Bidang 6 | **Lembaga Pembiayaan** | Perusahaan pembiayaan, modal ventura, LKM, LJK lainnya |
| Bidang 7 | **ITSK** | Inovasi teknologi sektor keuangan, aset keuangan digital, aset kripto |
| Bidang 8 | **PPEP** | Perilaku pelaku usaha jasa keuangan, edukasi, pelindungan konsumen |

Mengelompokkan peraturan secara manual ke departemen yang tepat membutuhkan waktu dan pemahaman konteks setiap regulasi. Project ini mengotomasi proses tersebut menggunakan machine learning berbasis teks.

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
train_model.py              # Training model TF-IDF + Multinomial NB
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

Pelabelan awal dilakukan secara rule-based menggunakan keyword matching. Setiap departemen memiliki daftar kata kunci yang mengacu pada Struktur Organisasi OJK-Wide. Dokumen diklasifikasikan berdasarkan skor kemunculan kata kunci, dengan frasa yang lebih panjang diberi bobot lebih tinggi untuk mengurangi ambiguitas.

### Model

Model yang digunakan adalah Multinomial Naive Bayes dengan TF-IDF vectorizer. Konfigurasi:
- **Vectorizer**: TfidfVectorizer, max_features=5000, ngram_range=(1,2), sublinear_tf=True
- **Classifier**: MultinomialNB, alpha=0.1
- Cross-validation 5-fold menghasilkan akurasi rata-rata sekitar 83.9%

### Dataset

- 43 dokumen POJK dari tahun 2024-2025
- Distribusi label: Perbankan (16), Pasar Modal (13), ITSK (5), Perasuransian (4), PPEP (3), Lembaga Pembiayaan (2)

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
| scikit-learn | 1.8.0 | TF-IDF, Naive Bayes, evaluasi model |
| joblib | 1.5.1 | Serialisasi model |
| pdfplumber | - | Ekstraksi teks dari PDF |
| pandas | - | Manipulasi data |

## Lisensi

MIT
