"""
Text Classification Model for OJK Regulations
================================================
Uses TF-IDF Vectorizer + Multinomial Naive Bayes to predict the
'Department' label (Perbankan, Pasar Modal, IKNB, ITSK) based on
document content.
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


# ── 1. Load Data ─────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the classified CSV into a DataFrame."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df)} documents from '{csv_path}'")
    print(f"\nLabel distribution:\n{df['department'].value_counts()}\n")
    return df


# ── 2. Build Pipeline ────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Create a TF-IDF + Multinomial Naive Bayes pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,       # top 5000 terms
            ngram_range=(1, 2),      # unigrams + bigrams
            sublinear_tf=True,       # apply log normalization
            min_df=1,
            max_df=0.95,
        )),
        ("clf", MultinomialNB(alpha=0.1)),  # smoothing parameter
    ])
    return pipeline


# ── 3. Train & Evaluate ──────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Train the model, run cross-validation, and evaluate on a test split."""
    X = df["content"]
    y = df["department"]

    # ── Cross-validation (on full dataset, useful for small datasets) ────
    pipeline = build_pipeline()
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print("=" * 60)
    print("5-Fold Cross-Validation")
    print("=" * 60)
    print(f"  Accuracy per fold : {cv_scores}")
    print(f"  Mean accuracy     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # ── Train/Test split for detailed report ─────────────────────────────
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

    # ── Re-train on full dataset for final model ─────────────────────────
    print("Re-training on full dataset for final model...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    return final_pipeline


# ── 4. Save Model ────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str) -> None:
    """Save the trained pipeline (TF-IDF + NB) to disk using joblib."""
    joblib.dump(pipeline, path)
    print(f"Model saved to '{path}'")


# ── 5. Load & Predict ────────────────────────────────────────────────────────

def load_model(path: str) -> Pipeline:
    """Load a previously saved model from disk."""
    pipeline = joblib.load(path)
    print(f"Model loaded from '{path}'")
    return pipeline


def predict(pipeline: Pipeline, texts: list[str]) -> list[str]:
    """Predict department labels for a list of texts."""
    return pipeline.predict(texts).tolist()


# ── Main ──────────────────────────────────────────────────────────────────────

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
