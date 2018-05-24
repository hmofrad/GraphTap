#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=hello.out
#SBATCH --error=hello.err
#SBATCH --ntasks=8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
# #SBATCH --sockets-per-node=2
# #SBATCH --cores-per-socket=8
#SBATCH --time=00:03:00
##SBATCH --cluster=smp
##SBATCH --partition=smp
#SBATCH --cluster=mpi
#SBATCH --partition=opa


export SLURM_CPU_BIND="none"
export I_MPI_FABRICS=shm:ofa

module purge
#module load intel
module load openmpi
module load gcc/6.3.0
echo "Starting mpirun"

NP=$SLURM_NTASKS
APP=proc2tile

mpirun  -np ${NP} ./${APP}

