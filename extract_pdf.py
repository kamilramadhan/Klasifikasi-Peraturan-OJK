import os
import re
import csv
import pdfplumber


def clean_text(text: str) -> str:
    """Hapus angka dan karakter khusus, sisakan huruf dan spasi."""
    # Hapus angka
    text = re.sub(r"\d+", "", text)
    # Hapus karakter khusus
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Gabungkan spasi berlebih
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Ekstrak seluruh teks dari file PDF menggunakan pdfplumber."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
    return "\n".join(full_text)


def main():
    input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs_POJK")
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_pojk.csv")

    pdf_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    )

    print(f"Found {len(pdf_files)} PDF file(s) in '{input_folder}'")

    rows = []
    for i, filename in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(input_folder, filename)
        print(f"[{i}/{len(pdf_files)}] Processing: {filename}")
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(raw_text)
            rows.append({"filename": filename, "content": cleaned})
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            rows.append({"filename": filename, "content": ""})

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "content"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Output saved to '{output_csv}' ({len(rows)} rows)")


if __name__ == "__main__":
    main()
