#!/bin/bash
#SBATCH --job-name=graphtap1
#SBATCH --output=graphtap1.out
#SBATCH --error=graphtap1.err
#SBATCH --ntasks=64
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
##SBATCH --mem=50000
##SBATCH --sockets-per-node=2
##SBATCH --ntasks-per-socket=8
#SBATCH --time=01:00:00
#SBATCH --cluster=mpi
#SBATCH --partition=opa-high-mem

## while true; do free -g; sleep 10; done
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
export LD_LIBRARY_PATH=/ihome/rmelhem/moh18/boost/boost_1_67_0/stage/lib:/ihome/rmelhem/moh18/boost:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NP=$SLURM_NTASKS
MPI="mpirun -np"
APPS_PERFIX="bin"
#APPS=("./deg")
APPS=("./pr" "./sssp" "./bfs" "./cc")

GRAPHS_PERFIX="/zfs1/cs3580_2017F/moh18/real"
GRAPHS=("live-journal-1.bin" "usa-roads-1.bin" "protein-k-mer-1.bin" "uk-2005-1.bin" "twitter-2010-1.bin" "uk-2006-06-1.bin" "uk-union-1.bin")
GRAPHS_W=("live-journal-1-w.bin0" "usa-roads-1-w.bin0" "protein-k-mer-1-w.bin0" "uk-2005-1-w.bin0" "twitter-2010-1-w.bin0" "uk-2006-06-1-w.bin0" "uk-union-1-w.bin0")
GRAPHS_V=("4847572" "23947348" "214005018" "39459926" "41652231" "80644903" "133633041")
SRC_V=("10010" "200" "2" "21049533" "23934133" "48018932" "26927947")

if [ "${SLURM_NNODES}" = "4" ]
then
    II=("0" "1" "2");
elif [ "$SLURM_NNODES" = "8" ]
then
    II=("3");
elif [ "$SLURM_NNODES" = "16" ]
then
    II=("4");
elif [ "$SLURM_NNODES" = "24" ]
then
    I=("5")
elif [ "$SLURM_NNODES" = "32" ]
then
    I=("6")
else
    echo "Invalid input graph"
    exit;     
fi


for I in "${II[@]}";
    do
    GRAPH_V=${GRAPHS_V[${I}]}
    SRC=${SRC_V[${I}]}
    ITERS=20
    NTIMES=("1")
    #NTIMES=("1" "2" "3");
    for A in "${APPS[@]}";
    do
        for N in "${NTIMES[@]}";
        do
            if [ "${A}" = "./pr" ]
            then
                ITER_OR_SRC=${ITERS}
            else
                ITER_OR_SRC=${SRC}
            fi
            
            if [ "${A}" = "./sssp" ]
            then
                GRAPH="${GRAPHS_W[${I}]}"    
            else
                GRAPH="${GRAPHS[${I}]}"
            fi    
            
            CMD="${MPI} ${NP} ${APPS_PERFIX}/${A} ${GRAPHS_PERFIX}/${GRAPH} ${GRAPH_V} ${ITER_OR_SRC}"
            echo "Command=${CMD}"
            ${CMD}
        done
    done
done

strings graphtap1.out | grep 'Execute time'
#strings graphtap1.out | grep 'Iterations'
#strings graphtap1.out | grep 'Value checksum'
#strings graphtap1.out | grep 'Reachable vertices'

