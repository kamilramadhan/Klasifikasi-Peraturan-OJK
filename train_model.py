"""
Klasifikasi Teks Peraturan OJK
Menggunakan TF-IDF Vectorizer dan Multinomial Naive Bayes untuk
memprediksi label departemen (Perbankan, Pasar Modal, IKNB, ITSK)
berdasarkan isi dokumen.
"""

import csv
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

csv.field_size_limit(sys.maxsize)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "output_pojk_classified.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_klasifikasi_ojk.joblib")


def load_data(csv_path: str) -> pd.DataFrame:
    """Muat CSV berlabel ke DataFrame."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df)} documents from '{csv_path}'")
    print(f"\nLabel distribution:\n{df['department'].value_counts()}\n")
    return df


def build_pipeline() -> Pipeline:
    """Buat pipeline TF-IDF + Multinomial Naive Bayes."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Training model, cross-validation, dan evaluasi pada test split."""
    X = df["content"]
    y = df["department"]

    pipeline = build_pipeline()
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print("=" * 60)
    print("5-Fold Cross-Validation")
    print("=" * 60)
    print(f"  Accuracy per fold : {cv_scores}")
    print(f"  Mean accuracy     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print("Classification Report (Test Set)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    print()

    print("Training ulang pada seluruh dataset untuk model final...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    return final_pipeline


def save_model(pipeline: Pipeline, path: str) -> None:
    """Simpan pipeline ke disk menggunakan joblib."""
    joblib.dump(pipeline, path)
    print(f"Model saved to '{path}'")


def load_model(path: str) -> Pipeline:
    """Muat model dari disk."""
    pipeline = joblib.load(path)
    print(f"Model loaded from '{path}'")
    return pipeline


def predict(pipeline: Pipeline, texts: list[str]) -> list[str]:
    """Prediksi label departemen untuk daftar teks."""
    return pipeline.predict(texts).tolist()


def main():
    # Load
    df = load_data(INPUT_CSV)

    # Train & evaluate
    pipeline = train_and_evaluate(df)

    # Save
    save_model(pipeline, MODEL_PATH)

    # Demo: predict on a few sample texts
    print("\n" + "=" * 60)
    print("Demo Predictions")
    print("=" * 60)
    samples = [
        "peraturan tentang bank umum syariah dan unit usaha syariah",
        "ketentuan mengenai reksa dana dan pasar modal",
        "penyelenggaraan inovasi teknologi sektor keuangan digital",
        "produk asuransi jiwa dan dana pensiun",
    ]
    preds = predict(pipeline, samples)
    for text, label in zip(samples, preds):
        print(f"  [{label:<12}] {text}")


if __name__ == "__main__":
    main()
