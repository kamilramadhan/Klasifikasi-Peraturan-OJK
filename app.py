import os
import re

import joblib
from flask import Flask, render_template, request

"""
Flask application untuk klasifikasi pengaduan OJK.
Memuat model yang sudah di-training dan menyajikan antarmuka web
untuk menerima input teks dan menampilkan hasil prediksi departemen.
"""

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_klasifikasi_ojk.joblib")
model = joblib.load(MODEL_PATH)

DEPT_INFO = {
    "Perbankan": {
        "icon": "🏦",
        "color": "#2196F3",
        "description": "Mengawasi dan mengatur industri perbankan, termasuk bank umum, BPR, dan bank syariah.",
    },
    "Pasar Modal": {
        "icon": "📈",
        "color": "#FF9800",
        "description": "Mengawasi pasar modal, termasuk saham, obligasi, reksa dana, dan perusahaan efek.",
    },
    "IKNB": {
        "icon": "🛡️",
        "color": "#4CAF50",
        "description": "Mengawasi industri keuangan non-bank: asuransi, dana pensiun, lembaga pembiayaan.",
    },
    "ITSK": {
        "icon": "💻",
        "color": "#E91E63",
        "description": "Mengawasi inovasi teknologi sektor keuangan, fintech, dan layanan digital.",
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
