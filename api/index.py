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
            cleaned = clean_text(user_input)
            prediction = model.predict([cleaned])[0]
            dept_info = DEPT_INFO.get(prediction, {})

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([cleaned])[0]
                confidence = round(max(proba) * 100, 1)

    return render_template(
        "index.html",
        prediction=prediction,
        dept_info=dept_info,
        user_input=user_input,
        confidence=confidence,
    )
