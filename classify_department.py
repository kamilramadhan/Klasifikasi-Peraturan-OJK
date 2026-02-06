import csv
import os
import re
import sys

csv.field_size_limit(sys.maxsize)


def classify_department(text: str) -> str:
    """Classify OJK regulation into a department based on keyword matching."""
    text_lower = text.lower()

    # Define keywords for each department
    departments = {
        "ITSK": [
            "teknologi", "fintech", "digital", "sandbox",
            "inovasi teknologi sektor keuangan",
        ],
        "Pasar Modal": [
            "saham", "efek", "emiten", "reksa dana",
            "reksadana", "pasar modal", "obligasi daerah", "sukuk daerah",
        ],
        "IKNB": [
            "asuransi", "pensiun", "pembiayaan",
            "dana pensiun", "penjaminan", "lembaga pembiayaan",
        ],
        "Perbankan": [
            "bank", "bpr", "bprs", "syariah", "perbankan",
            "bank umum", "bank perekonomian rakyat",
        ],
    }

    # Count keyword hits per department
    scores: dict[str, int] = {}
    for dept, keywords in departments.items():
        count = 0
        for kw in keywords:
            count += len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
        scores[dept] = count

    # Return department with highest score, or 'Lainnya' if no match
    best_dept = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_dept] == 0:
        return "Lainnya"
    return best_dept


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "output_pojk.csv")
    output_csv = os.path.join(base_dir, "output_pojk_classified.csv")

    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dept = classify_department(row["content"])
            row["department"] = dept
            rows.append(row)

    # Write output with new 'department' column
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "content", "department"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"Classified {len(rows)} regulations → '{output_csv}'\n")
    print(f"{'Department':<15} {'Count':>5}")
    print("-" * 22)
    dept_counts: dict[str, int] = {}
    for row in rows:
        d = row["department"]
        dept_counts[d] = dept_counts.get(d, 0) + 1
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"{dept:<15} {count:>5}")

    # Show per-file classification
    print(f"\n{'No':<4} {'Filename':<75} {'Department'}")
    print("-" * 100)
    for i, row in enumerate(rows, 1):
        print(f"{i:<4} {row['filename']:<75} {row['department']}")


if __name__ == "__main__":
    main()
