#!/bin/bash
#SBATCH --job-name=la3
#SBATCH --output=la3.out
#SBATCH --error=la3.err
#SBATCH --ntasks=128
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
##SBATCH --sockets-per-node=2
##SBATCH --ntasks-per-socket=8
#SBATCH --time=01:00:00
#SBATCH --cluster=mpi
#SBATCH --partition=opa

##SSBATCH ---ntasks-per-core=1
##SBATCH --distribution cyclic:cyclic
##SSBATCH ---cores-per-socket=14



echo "SLURM_JOB_ID="$SLURM_JOB_ID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_CORES_NODES"=$SLURM_CPUS_PER_TASK
echo "SLURM_TASKS"=$SLURM_NTASKS
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "************************************************"

MPI="intel"
#MPI="openmpi"
if [ "${MPI}" = "mpich" ]
then
        echo "MPI"=${MPI}
        module purge
        module load  gcc/5.4.0
        module load mpich/3.1
else
        echo "MPI"=${MPI}
        module purge
        module load gcc/5.4.0
        module load intel
fi

#ulimit -s unlimited
#cd $SLURM_SCRATCH


export SLURM_CPU_BIND="none"
#export I_MPI_FABRICS_LIST="ofi,tmi" # OPA
export I_MPI_FABRICS=shm:ofa
export I_MPI_FALLBACK=0
export I_MPI_DEBUG=2
#exort I_MPI_HYDRA_DEBUG=on
export LD_LIBRARY_PATH=/ihome/rmelhem/moh18/boost/boost_1_67_0/stage/lib:/ihome/rmelhem/moh18/boost:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NP=$SLURM_NTASKS
APP=./main

if [ "$SLURM_NNODES" = "4" ]
then
        echo "RMAT=26"
        time mpirun -np ${NP} ${APP} /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864 20
elif [ "$SLURM_NNODES" = "8" ]
then
        echo "RMAT=27"
        time mpirun -np ${NP} ${APP}  /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728 20
elif [ "$SLURM_NNODES" = "16" ]
then
        echo "RMAT=28"
        time mpirun -np ${NP} ${APP} /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28.bin 268435456 20
else
        echo "RMAT=$SLURM_NNODES N/A"
fi
