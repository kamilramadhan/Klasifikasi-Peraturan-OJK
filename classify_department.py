import csv
import os
import re
import sys

csv.field_size_limit(sys.maxsize)


def classify_department(text: str) -> str:
    """Klasifikasikan peraturan OJK ke departemen berdasarkan keyword matching.

    Mengacu pada Struktur Organisasi OJK-Wide (Eksisting):
    - Bidang 3: Pengawas Perbankan
    - Bidang 4: Pengawas Pasar Modal, Keuangan Derivatif, dan Bursa Karbon
    - Bidang 5: Pengawas Perasuransian, Penjaminan, dan Dana Pensiun
    - Bidang 6: Pengawas Lembaga Pembiayaan, PMV, LKM, dan LJK Lainnya
    - Bidang 7: Pengawas ITSK, Aset Keuangan Digital, dan Aset Kripto
    - Bidang 8: Pengawas Perilaku Pelaku Usaha Jasa Keuangan, Edukasi,
                dan Pelindungan Konsumen (PPEP)
    """
    text_lower = text.lower()

    # Kata kunci per departemen, urut dari yang paling spesifik
    departments = {
        "ITSK": [
            "inovasi teknologi sektor keuangan",
            "aset keuangan digital", "aset kripto",
            "fintech", "sandbox regulatori", "sandbox",
            "regulatory sandbox",
            "teknologi finansial", "agregasi jasa keuangan",
            "urun dana", "layanan urun dana",
            "penyelenggara sistem elektronik",
            "penyelenggara inovasi",
            "aset digital",
            "perdagangan aset keuangan digital",
            "penilaian kemampuan dan kepatutan",
        ],
        "PPEP": [
            "pelindungan konsumen", "perlindungan konsumen",
            "edukasi keuangan", "literasi keuangan",
            "perilaku pelaku usaha", "inklusi keuangan",
            "pengaduan konsumen", "penanganan pengaduan",
            "satuan tugas penanganan",
            "usaha tanpa izin",
            "pungutan di sektor jasa keuangan",
            "profesi penunjang",
        ],
        "Perasuransian": [
            "perusahaan perasuransian",
            "perasuransian",
            "asuransi jiwa", "asuransi umum", "asuransi syariah",
            "produk asuransi", "reasuransi",
            "dana pensiun", "program pensiun",
            "lembaga penjamin", "penjaminan",
        ],
        "Lembaga Pembiayaan": [
            "lembaga pembiayaan",
            "perusahaan pembiayaan", "pembiayaan syariah",
            "modal ventura", "perusahaan modal ventura",
            "lembaga keuangan mikro",
            "sarana multi infrastruktur",
            "konglomerasi keuangan", "perusahaan induk",
            "kemudahan akses pembiayaan",
            "usaha mikro kecil", "umkm",
        ],
        "Pasar Modal": [
            "pasar modal",
            "efek bersifat ekuitas", "efek bersifat utang",
            "perusahaan efek", "penjamin emisi efek",
            "perantara pedagang efek",
            "reksa dana", "reksadana", "manajer investasi",
            "emiten", "perusahaan terbuka",
            "obligasi daerah", "sukuk daerah",
            "efek syariah", "daftar efek syariah",
            "bursa karbon", "derivatif keuangan",
            "kustodian", "agen penjual efek",
            "rapat umum pemegang saham",
            "laporan kepemilikan",
            "penyedia likuiditas",
            "dematerialisasi efek",
        ],
        "Perbankan": [
            "bank umum", "bank perekonomian rakyat",
            "bank syariah", "unit usaha syariah",
            "bpr", "bprs",
            "perbankan",
            "suku bunga dasar kredit",
            "rasio kecukupan likuiditas", "liquidity coverage",
            "rasio pendanaan stabil", "net stable funding",
            "rasio pengungkit",
            "rahasia bank",
            "laporan bank", "integritas pelaporan keuangan bank",
            "slik", "sistem layanan informasi keuangan",
            "kualitas aset bank",
            "tata kelola bagi bpr",
            "perluasan kegiatan usaha perbankan",
            "anti fraud",
            "transparansi dan publikasi",
            "status pengawasan",
        ],
    }

    # Hitung kemunculan keyword per departemen (frasa panjang diberi bobot lebih)
    scores: dict[str, int] = {}
    for dept, keywords in departments.items():
        count = 0
        for kw in keywords:
            matches = len(re.findall(re.escape(kw), text_lower))
            word_count = len(kw.split())
            count += matches * word_count
        scores[dept] = count

    # Departemen dengan skor tertinggi
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

    # Tulis output dengan kolom 'department'
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "content", "department"])
        writer.writeheader()
        writer.writerows(rows)

    # Ringkasan
    print(f"Classified {len(rows)} regulations -> '{output_csv}'\n")
    print(f"{'Department':<25} {'Count':>5}")
    print("-" * 32)
    dept_counts: dict[str, int] = {}
    for row in rows:
        d = row["department"]
        dept_counts[d] = dept_counts.get(d, 0) + 1
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"{dept:<25} {count:>5}")

    # Klasifikasi per file
    print(f"\n{'No':<4} {'Filename':<75} {'Department'}")
    print("-" * 100)
    for i, row in enumerate(rows, 1):
        print(f"{i:<4} {row['filename']:<75} {row['department']}")


if __name__ == "__main__":
    main()
