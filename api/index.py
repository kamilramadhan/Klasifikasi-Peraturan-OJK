import os
import re

import joblib
from flask import Flask, render_template, request

"""
Entry point serverless untuk deployment Vercel.
Identik dengan app.py, dengan penyesuaian path karena file
berada di subdirektori api/.
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

MODEL_PATH = os.path.join(BASE_DIR, "model_klasifikasi_ojk.joblib")
model = joblib.load(MODEL_PATH)

DEPT_INFO = {
    "Perbankan": {
        "icon": "BNK",
        "color": "#1565C0",
        "description": "Pengawasan bank umum, BPR, BPRS, bank syariah, dan unit usaha syariah.",
    },
    "Pasar Modal": {
        "icon": "PM",
        "color": "#E65100",
        "description": "Pengawasan pasar modal, keuangan derivatif, dan bursa karbon.",
    },
    "Perasuransian": {
        "icon": "ASR",
        "color": "#2E7D32",
        "description": "Pengawasan perasuransian, penjaminan, dan dana pensiun.",
    },
    "Lembaga Pembiayaan": {
        "icon": "LP",
        "color": "#6A1B9A",
        "description": "Pengawasan lembaga pembiayaan, perusahaan modal ventura, LKM, dan LJK lainnya.",
    },
    "ITSK": {
        "icon": "IT",
        "color": "#C62828",
        "description": "Pengawasan inovasi teknologi sektor keuangan, aset keuangan digital, dan aset kripto.",
    },
    "PPEP": {
        "icon": "PP",
        "color": "#00838F",
        "description": "Pengawasan perilaku pelaku usaha jasa keuangan, edukasi, dan pelindungan konsumen.",
    },
}

SYNONYM_MAP = {
    "qris": "inovasi teknologi sektor keuangan sistem pembayaran",
    "ewallet": "inovasi teknologi sektor keuangan layanan keuangan digital",
    "e-wallet": "inovasi teknologi sektor keuangan layanan keuangan digital",
    "dompet digital": "inovasi teknologi sektor keuangan layanan keuangan digital",
    "gopay": "inovasi teknologi sektor keuangan layanan keuangan digital",
    "ovo": "inovasi teknologi sektor keuangan layanan keuangan digital",
    "dana": "inovasi teknologi sektor keuangan layanan keuangan digital",
    "shopeepay": "inovasi teknologi sektor keuangan layanan keuangan digital",
    "paylater": "inovasi teknologi sektor keuangan layanan pinjam meminjam",
    "pay later": "inovasi teknologi sektor keuangan layanan pinjam meminjam",
    "pinjol": "inovasi teknologi sektor keuangan layanan pinjam meminjam",
    "pinjaman online": "inovasi teknologi sektor keuangan layanan pinjam meminjam",
    "p2p lending": "inovasi teknologi sektor keuangan layanan pinjam meminjam",
    "peer to peer": "inovasi teknologi sektor keuangan layanan pinjam meminjam",
    "crowdfunding": "inovasi teknologi sektor keuangan layanan urun dana",
    "bitcoin": "aset keuangan digital aset kripto",
    "crypto": "aset keuangan digital aset kripto",
    "kripto": "aset keuangan digital aset kripto",
    "cryptocurrency": "aset keuangan digital aset kripto",
    "blockchain": "aset keuangan digital inovasi teknologi",
    "nft": "aset keuangan digital aset kripto",
    "token digital": "aset keuangan digital aset kripto",
    "robo advisor": "inovasi teknologi sektor keuangan manajer investasi",
    "penipuan": "pelindungan konsumen perilaku pelaku usaha",
    "ditipu": "pelindungan konsumen perilaku pelaku usaha",
    "scam": "pelindungan konsumen perilaku pelaku usaha",
    "komplain": "pengaduan konsumen pelindungan konsumen",
    "pengaduan": "pengaduan konsumen pelindungan konsumen",
    "aduan": "pengaduan konsumen pelindungan konsumen",
    "lapor": "pengaduan konsumen pelindungan konsumen",
    "rugikan konsumen": "pelindungan konsumen perilaku pelaku usaha",
    "investasi bodong": "pelindungan konsumen usaha tanpa izin",
    "ilegal": "pelindungan konsumen usaha tanpa izin",
    "tabungan": "bank umum perbankan",
    "deposito": "bank umum perbankan",
    "kredit macet": "bank umum kualitas aset perbankan",
    "kpr": "bank umum perbankan kredit",
    "atm": "bank umum perbankan",
    "kartu kredit": "bank umum perbankan",
    "mobile banking": "bank umum perbankan",
    "internet banking": "bank umum perbankan",
    "investasi saham": "pasar modal efek emiten",
    "reksadana": "reksa dana pasar modal manajer investasi",
    "broker saham": "perantara pedagang efek pasar modal",
    "ipo": "penjamin emisi efek pasar modal emiten",
    "obligasi": "efek bersifat utang pasar modal",
    "dividen": "emiten pasar modal efek",
    "klaim asuransi": "perusahaan perasuransian produk asuransi",
    "premi asuransi": "perusahaan perasuransian produk asuransi",
    "bpjs": "perusahaan perasuransian asuransi",
    "jiwasraya": "perusahaan perasuransian asuransi jiwa",
    "unit link": "perusahaan perasuransian produk asuransi",
    "unitlink": "perusahaan perasuransian produk asuransi",
    "leasing": "lembaga pembiayaan perusahaan pembiayaan",
    "kredit motor": "lembaga pembiayaan perusahaan pembiayaan",
    "kredit mobil": "lembaga pembiayaan perusahaan pembiayaan",
    "sewa guna usaha": "lembaga pembiayaan perusahaan pembiayaan",
    "multifinance": "lembaga pembiayaan perusahaan pembiayaan",
    "gadai": "lembaga pembiayaan lembaga keuangan mikro",
    "koperasi simpan pinjam": "lembaga pembiayaan lembaga keuangan mikro",
}


def expand_synonyms(text: str) -> str:
    """Ganti istilah populer dengan terminologi formal peraturan OJK."""
    text_lower = text.lower()
    for colloquial, formal in SYNONYM_MAP.items():
        text_lower = re.sub(
            r"\b" + re.escape(colloquial) + r"\b",
            formal,
            text_lower,
        )
    return text_lower


KEYWORD_OVERRIDE = {
    "Lembaga Pembiayaan": [
        "leasing", "multifinance", "sewa guna usaha",
        "kredit motor", "kredit mobil", "gadai",
        "koperasi simpan pinjam", "modal ventura",
    ],
}


def keyword_override(text: str, model_pred: str, model_conf: float) -> str:
    """Override prediksi model jika confidence rendah dan ada keyword kuat."""
    if model_conf > 70.0:
        return model_pred
    text_lower = text.lower()
    for dept, keywords in KEYWORD_OVERRIDE.items():
        for kw in keywords:
            if kw in text_lower:
                return dept
    return model_pred


def clean_text(text: str) -> str:
    """Hapus angka dan karakter khusus, sisakan huruf dan spasi."""
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    dept_info = None
    user_input = ""
    confidence = None

    if request.method == "POST":
        user_input = request.form.get("complaint", "").strip()
        if user_input:
            expanded = expand_synonyms(user_input)
            cleaned = clean_text(expanded)
            prediction = model.predict([cleaned])[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([cleaned])[0]
                confidence = round(max(proba) * 100, 1)
            else:
                confidence = None

            prediction = keyword_override(user_input, prediction, confidence or 100.0)
            dept_info = DEPT_INFO.get(prediction, {})

    return render_template(
        "index.html",
        prediction=prediction,
        dept_info=dept_info,
        user_input=user_input,
        confidence=confidence,
    )
