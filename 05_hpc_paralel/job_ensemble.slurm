#!/bin/bash
# Templat batch script Slurm (belum diuji pada klaster tertentu; sesuaikan nama partisi/modul dengan klaster Anda).
#SBATCH --job-name=ensemble-banjir
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --output=ensemble_%j.log

module load python openmpi      # nama modul berbeda-beda per klaster
source venv/bin/activate        # python -m venv venv && pip install numpy mpi4py

mpirun -n $SLURM_NTASKS python mpi_ensemble.py
