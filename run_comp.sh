#!/bin/bash -x
#SBATCH -J comp         # Job name
#SBATCH -o comp.%j.out     # Name of stdout file (%j expands to jobID)
##SBATCH -p rsa          # Queue name 
##SBATCH -N 4                   # Total number of nodes 16 cores per node
#SBATCH  -n 8                 # Total number of MPI tasks requested 
##SBATCH -t 01:00:00
##SBATCH --ntasks-per-node=8
##SBATCH --exclusive

# File paths
HOSTFILE=/tmp/hosts.$SLURM_JOB_ID
MPIRUN=/usr/local/openmpi/1.6.5-ib/bin/mpirun
PYTHON=/fasttmp/mengp2/HYDRA/python/2.7.10/bin/python
TARGET=./kona_compst.py

# Output directory
OUTDIR="./L2_compst" #"./OUTPUT/RSA_L1"

# Mesh size specification: L0 - 850k | L1 - 100k | L2 - 15k
MESH="L2"

# Number of shape design variables: 72, 192 or 768)
NDV=72

# Max number of optimization iterations
ITER=20

# EQU=False

MUINIT=0.3
MUPOW=0.3
RINIT=0.05
RMAX=2

srun hostname -s > $HOSTFILE

if [ -z "$SLURM_NPROCS" ] ; then
  if [ -z "$SLURM_NTASKS_PER_NODE" ] ; then
    SLURM_NTASKS_PER_NODE=1
  fi
  SLURM_NPROCS=$(( $SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE ))
fi

$MPIRUN -machinefile $HOSTFILE -np $SLURM_NPROCS $PYTHON $TARGET \
    --output=$OUTDIR --mesh=$MESH --FFD=$NDV \
    --iter=$ITER  --muinit=$MUINIT  --mupow=$MUPOW \
    --rinit=$RINIT --rmax=$RMAX

rm /tmp/hosts.$SLURM_JOB_ID
