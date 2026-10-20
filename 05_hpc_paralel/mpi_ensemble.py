"""
Versi MPI dari ensemble banjir untuk klaster HPC (membutuhkan mpi4py + implementasi MPI).

CATATAN: skrip ini disertakan sebagai templat untuk klaster. Di lingkungan pembuatan repo ini
mpi4py tidak terpasang sehingga skrip ini BELUM diuji; versi multiprocessing
(hpc_banjir_ensemble.py) yang sudah diuji.

Jalankan:  mpirun -n 4 python mpi_ensemble.py
"""
import numpy as np
from mpi4py import MPI

from hpc_banjir_ensemble import AMBANG_BANJIR, simulasi_skenario

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
N = 2000

seeds = np.array_split(np.arange(N), size)[rank]          # bagi skenario antar-rank
lokal = np.array([simulasi_skenario(int(s)) for s in seeds])
semua = comm.gather(lokal, root=0)                         # kumpulkan hasil di rank 0

if rank == 0:
    puncak = np.concatenate(semua)
    print(f"{N} skenario di {size} rank; P(banjir) = {(puncak > AMBANG_BANJIR).mean():.1%}")
