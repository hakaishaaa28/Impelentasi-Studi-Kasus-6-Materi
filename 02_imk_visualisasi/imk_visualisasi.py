"""
Studi Kasus 2 - Human Computer Interaction (IMK) dan Visualisasi Data
=====================================================================
Mengubah prinsip materi menjadi kode yang bisa diuji:
  A. Chartjunk vs desain bersih (data-ink ratio) untuk data yang sama
  B. Beban kognitif: pie chart 9 kategori vs bar chart terurut
  C. Prinsip Gestalt (proximity, similarity, enclosure) pada tata letak data
  D. Evaluator heuristik Nielsen (10 heuristik) -> skor & laporan CSV

Jalankan:  python imk_visualisasi.py
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Data: waktu rata-rata (detik) pengguna menyelesaikan tugas per fitur aplikasi
FITUR = ["Login", "Cari produk", "Filter", "Keranjang", "Alamat", "Pembayaran",
         "Voucher", "Lacak pesanan", "Ulasan"]
DETIK = np.array([14, 38, 52, 21, 47, 66, 33, 18, 29])


def demo_chartjunk():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    # buruk: 3D-ish, warna ramai, grid tebal, tanpa label langsung
    ax = axs[0]
    ax.bar(FITUR, DETIK, color=plt.cm.rainbow(np.linspace(0, 1, len(FITUR))),
           edgecolor="black", linewidth=2, hatch="//")
    ax.grid(True, linewidth=2, color="black", alpha=0.6)
    ax.set_facecolor("#ffe9a8")
    ax.set_title("BURUK: chartjunk", fontweight="bold", color="red")
    ax.tick_params(axis="x", rotation=90)
    # baik: satu warna, penekanan pada outlier, label langsung
    ax = axs[1]
    order = np.argsort(DETIK)
    names, vals = np.array(FITUR)[order], DETIK[order]
    colors = ["#c0392b" if v == vals.max() else "#9aa5b1" for v in vals]
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v}s", va="center", fontsize=9)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.set_title("BAIK: terurut, satu warna sorot, label langsung\nPembayaran paling lambat (66 detik)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "A_chartjunk_vs_bersih.png"), dpi=140)
    plt.close(fig)


def demo_beban_kognitif():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].pie(DETIK, labels=FITUR, autopct="%1.0f%%", textprops={"fontsize": 7})
    axs[0].set_title("Pie 9 irisan: sulit dibandingkan")
    order = np.argsort(DETIK)[::-1]
    axs[1].bar(np.array(FITUR)[order], DETIK[order], color="#2c7fb8")
    axs[1].tick_params(axis="x", rotation=45)
    axs[1].spines[["top", "right"]].set_visible(False)
    axs[1].set_ylabel("Detik")
    axs[1].set_title("Bar terurut: perbandingan langsung")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "B_beban_kognitif.png"), dpi=140)
    plt.close(fig)


def demo_gestalt():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    rng = np.random.default_rng(0)
    # Proximity: jarak menentukan kelompok
    pts = np.vstack([rng.normal([1, 1], 0.25, (12, 2)), rng.normal([3.5, 1], 0.25, (12, 2))])
    axs[0].scatter(*pts.T, s=60, color="#34495e")
    axs[0].set_title("Proximity: dua kelompok terlihat")
    # Similarity: warna sama = kategori sama
    xs, ys = np.meshgrid(range(6), range(4))
    cat = (xs + ys) % 2
    axs[1].scatter(xs.ravel(), ys.ravel(), s=120, c=np.where(cat.ravel() == 0, "#e67e22", "#2980b9"))
    axs[1].set_title("Similarity: warna = kategori")
    # Enclosure: kotak membungkus elemen terkait
    for i, (label, x) in enumerate([("Info akun", 0.3), ("Pengiriman", 3.6)]):
        axs[2].add_patch(FancyBboxPatch((x, 0.4), 3, 2.2, boxstyle="round,pad=0.05", fc="#ecf0f1", ec="#7f8c8d"))
        axs[2].text(x + 0.15, 2.3, label, fontweight="bold", fontsize=9)
        for j in range(3):
            axs[2].text(x + 0.2, 1.7 - j * 0.5, f"- field {j + 1}", fontsize=9)
    axs[2].set_xlim(0, 7)
    axs[2].set_ylim(0, 3)
    axs[2].set_title("Enclosure: kotak mengelompokkan")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "C_gestalt.png"), dpi=140)
    plt.close(fig)


# ---------- Evaluator heuristik Nielsen ----------
HEURISTIK = [
    "Visibilitas status sistem",
    "Kesesuaian sistem dengan dunia nyata",
    "Kontrol dan kebebasan pengguna",
    "Konsistensi dan standar",
    "Pencegahan kesalahan",
    "Mengenali daripada mengingat",
    "Fleksibilitas dan efisiensi",
    "Desain estetis dan minimalis",
    "Bantu pengguna mengenali & pulih dari error",
    "Bantuan dan dokumentasi",
]


def evaluasi_nielsen(skor_per_evaluator):
    """skor_per_evaluator: array (n_evaluator, 10), skala keparahan 0 (aman) - 4 (katastrofik)."""
    s = np.asarray(skor_per_evaluator, float)
    rata = s.mean(axis=0)
    rows = []
    for h, r in zip(HEURISTIK, rata):
        prioritas = "KRITIS" if r >= 3 else "TINGGI" if r >= 2 else "SEDANG" if r >= 1 else "RENDAH"
        rows.append((h, round(float(r), 2), prioritas))
    return sorted(rows, key=lambda t: -t[1])


def demo_nielsen():
    # 3 evaluator mengamati prototipe checkout (data contoh)
    skor = [[1, 0, 3, 1, 4, 2, 2, 1, 3, 1],
            [2, 1, 3, 0, 3, 2, 1, 2, 4, 0],
            [1, 0, 2, 1, 4, 1, 2, 1, 3, 1]]
    hasil = evaluasi_nielsen(skor)
    path = os.path.join(OUT, "D_laporan_heuristik.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["heuristik", "rata_keparahan", "prioritas"])
        w.writerows(hasil)
    print("Laporan evaluasi heuristik Nielsen (urut keparahan):")
    for h, r, p in hasil[:5]:
        print(f"  [{p:<6}] {r:>4}  {h}")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [h for h, _, _ in hasil][::-1]
    vals = [r for _, r, _ in hasil][::-1]
    ax.barh(names, vals, color=["#c0392b" if v >= 3 else "#9aa5b1" for v in vals])
    ax.set_xlabel("Rata-rata keparahan (0-4)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Temuan evaluasi heuristik prototipe checkout")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D_heuristik.png"), dpi=140)
    plt.close(fig)


def main():
    argparse.ArgumentParser().parse_known_args()  # terima --fast dari run_all
    os.makedirs(OUT, exist_ok=True)
    demo_chartjunk()
    demo_beban_kognitif()
    demo_gestalt()
    demo_nielsen()
    print(f"\n4 grafik + 1 CSV disimpan di: {OUT}")
    print("Ingat: AI/otomasi mempercepat desain, tetapi empati & keputusan akhir tetap di manusia.")


if __name__ == "__main__":
    main()
