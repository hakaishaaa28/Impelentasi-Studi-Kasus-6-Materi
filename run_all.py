"""Menjalankan keenam studi kasus berurutan. Pemakaian: python run_all.py [--fast]"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
KASUS = [
    ("01_deforestasi_ai", "deforestasi.py"),
    ("02_imk_visualisasi", "imk_visualisasi.py"),
    ("03_machine_learning", "ml_pipeline.py"),
    ("04_analitik_crispdm", "crispdm_churn.py"),
    ("05_hpc_paralel", "hpc_banjir_ensemble.py"),
    ("06_gis_spasial", "gis_analisis.py"),
]

gagal = []
for folder, script in KASUS:
    print(f"\n##### {folder}/{script} #####", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, script] + sys.argv[1:], cwd=ROOT / folder)
    print(f"[{'OK' if r.returncode == 0 else 'GAGAL'}] {time.time() - t0:.1f} detik")
    if r.returncode:
        gagal.append(folder)

print("\nRingkasan:", "semua berhasil" if not gagal else f"gagal: {gagal}")
sys.exit(1 if gagal else 0)
