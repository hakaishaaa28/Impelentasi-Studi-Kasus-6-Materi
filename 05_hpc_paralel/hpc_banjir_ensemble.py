"""
Studi Kasus 5 - Menjembatani Infrastruktur HPC untuk Masalah Lokal
==================================================================
Masalah lokal: estimasi risiko banjir sungai dari ketidakpastian curah hujan.
Satu simulasi = satu skenario hujan; risiko diperoleh dari RIBUAN skenario (ensemble Monte Carlo).
Ini contoh "embarrassingly parallel": tiap skenario independen sehingga cocok untuk HPC.

Skrip ini:
  1. Menjalankan ensemble secara serial dan paralel (multiprocessing) lalu mengukur speedup.
  2. Membandingkan dengan hukum Amdahl.
  3. Memastikan hasil serial == paralel (reproducibility via seed per-skenario).

Di klaster nyata, pola yang sama dijalankan dengan MPI atau job array Slurm
(lihat job_ensemble.slurm dan mpi_ensemble.py).

Jalankan:  python hpc_banjir_ensemble.py [--fast]
"""
import argparse
import os
import time
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
N_CELLS, N_STEPS = 200, 1500
AMBANG_BANJIR = 6.0  # meter di outlet (nilai ilustrasi)


def simulasi_skenario(seed):
    """Routing aliran 1D sederhana (advection + dissipasi) untuk satu skenario hujan. Mengembalikan puncak muka air outlet."""
    rng = np.random.default_rng(seed)
    intensitas = rng.gamma(2.0, 1.2)           # mm/jam, ketidakpastian hujan
    durasi = rng.integers(100, 600)             # langkah waktu hujan
    h = np.zeros(N_CELLS)
    puncak = 0.0
    for t in range(N_STEPS):
        hujan = intensitas * 0.004 if t < durasi else 0.0
        h += hujan * (1 + 0.5 * np.sin(np.linspace(0, 3, N_CELLS)))   # hujan tidak seragam di DAS
        aliran = 0.25 * h
        h = h - aliran + np.concatenate(([0.0], aliran[:-1]))          # air bergerak ke hilir
        h *= 0.999                                                     # infiltrasi
        puncak = max(puncak, h[-1])
    return puncak


def jalankan(n, workers):
    seeds = list(range(n))
    t0 = time.perf_counter()
    if workers == 1:
        hasil = [simulasi_skenario(s) for s in seeds]
    else:
        with Pool(workers) as p:
            hasil = p.map(simulasi_skenario, seeds, chunksize=max(1, n // (workers * 4)))
    return np.array(hasil), time.perf_counter() - t0


def amdahl(p, n):
    return 1 / ((1 - p) + p / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a, _ = ap.parse_known_args()
    os.makedirs(OUT, exist_ok=True)
    n = 60 if a.fast else 240
    maks = max(2, min(cpu_count(), 8))
    worker_list = sorted({1, 2, maks})
    print(f"Ensemble {n} skenario, CPU tersedia: {cpu_count()}, uji worker: {worker_list}")

    waktu, ref = {}, None
    for w in worker_list:
        hasil, dt = jalankan(n, w)
        waktu[w] = dt
        if ref is None:
            ref = hasil
        sama = np.allclose(ref, hasil)
        print(f"  worker={w}: {dt:6.2f} s  speedup={waktu[1] / dt:4.2f}x  hasil identik dengan serial: {sama}")
        assert sama, "hasil paralel harus identik (seed per-skenario)"

    prob = float((ref > AMBANG_BANJIR).mean())
    print(f"\nRisiko: P(puncak outlet > {AMBANG_BANJIR} m) = {prob:.1%}  | puncak median = {np.median(ref):.2f} m, P95 = {np.percentile(ref, 95):.2f} m")

    # perkirakan fraksi paralel dari data terukur (pakai worker terbesar)
    wmax = max(worker_list)
    s_nyata = waktu[1] / waktu[wmax]
    p_est = min(1.0, (1 - 1 / s_nyata) / (1 - 1 / wmax)) if wmax > 1 and s_nyata > 1 else 0.0
    print(f"Fraksi paralel terestimasi (Amdahl): {p_est:.2f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(ref, bins=25, color="#2c7fb8")
    axs[0].axvline(AMBANG_BANJIR, color="red", ls="--", label="ambang banjir")
    axs[0].set_xlabel("Puncak muka air outlet (m)")
    axs[0].set_ylabel("Jumlah skenario")
    axs[0].set_title(f"Distribusi risiko (P banjir = {prob:.0%})")
    axs[0].legend()
    ns = np.arange(1, 65)
    axs[1].plot(ns, amdahl(0.95, ns), label="Amdahl p=0.95")
    axs[1].plot(ns, amdahl(0.99, ns), label="Amdahl p=0.99")
    axs[1].scatter(worker_list, [waktu[1] / waktu[w] for w in worker_list], color="red", zorder=5, label="terukur di mesin ini")
    axs[1].set_xlabel("Jumlah core")
    axs[1].set_ylabel("Speedup")
    axs[1].set_title("Speedup dibatasi bagian serial")
    axs[1].legend()
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "risiko_dan_speedup.png"), dpi=140)
    print(f"\nGrafik disimpan di: {OUT}")


if __name__ == "__main__":
    main()
