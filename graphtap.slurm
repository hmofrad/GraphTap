#!/bin/bash
#SBATCH --job-name=graphtap
#SBATCH --output=graphtap.out
#SBATCH --error=graphtap.err
#SBATCH --ntasks=16
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
##SBATCH --mem=50000
##SBATCH --sockets-per-node=2
##SBATCH --ntasks-per-socket=8
#SBATCH --time=00:10:00
#SBATCH --cluster=mpi
#SBATCH --partition=ib

## while true; do crc-squeue.py; sleep 10; done
## grep 'Execution' la3.out
## strings graphtap.out | grep 'Execute'


echo "SLURM_JOB_ID="$SLURM_JOB_ID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_CORES_NODES"=$SLURM_CPUS_PER_TASK
echo "SLURM_TASKS"=$SLURM_NTASKS
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "************************************************"

module purge
module load gcc/5.4.0
module load intel
export I_MPI_FABRICS=shm:ofa

export SLURM_CPU_BIND="none"
export I_MPI_FALLBACK=0
export I_MPI_DEBUG=2
export LD_LIBRARY_PATH=/ihome/rmelhem/moh18/boost/boost_1_67_0/stage/lib:/ihome/rmelhem/moh18/boost:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NP=$SLURM_NTASKS
APP=bin/./pr
ITERS_OR_SRC=20

if [ "$SLURM_NNODES" = "4" ]
then
    echo "RMAT=26"
    if [ "${APP}" = "sssp" ]
    then
        mpirun -np ${NP} ${APP}  /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26_1_w.bin 67108864 ${ITERS_OR_SRC}
    else
        mpirun -np ${NP} ${APP}  /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26_1.bin 67108864 ${ITERS_OR_SRC}
    fi
elif [ "$SLURM_NNODES" = "8" ]
then
    echo "RMAT=27"
    if [ "${APP}" = "sssp" ]
    then
        mpirun -np ${NP} ${APP}  /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27_1_w.bin 134217728 ${ITERS_OR_SRC}
    else
        mpirun -np ${NP} ${APP}  /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27_1.bin 134217728 ${ITERS_OR_SRC}
    fi    
elif [ "$SLURM_NNODES" = "16" ]
then
    echo "RMAT=28"
    if [ "${APP}" = "sssp" ]
    then
        mpirun -np ${NP} ${APP}  /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28_1_w.bin 268435456 ${ITERS_OR_SRC}
    else
        mpirun -np ${NP} ${APP}  /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28_1.bin 268435456 ${ITERS_OR_SRC}
    fi    
else
        echo "RMAT=$SLURM_NNODES N/A"
fi

strings graphtap.out | grep 'Execute'