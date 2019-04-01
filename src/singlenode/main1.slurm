#!/bin/bash
#SBATCH --job-name=main1
#SBATCH --output=main1.out
#SBATCH --error=main1.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --cluster=smp
#SBATCH --partition=high-mem

## while true; do free -g; sleep 10; done
## grep 'Execution' la3.out
## strings graphtap.out | grep 'Execute'
##

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

GRAPHS_PERFIX="/zfs1/cs3580_2017F/moh18/real"
GRAPHS=("uk-2005-1.bin-hashed.bin 39459926" "it-2004-1.bin0-hashed.bin 41291596" "twitter-2010-1.bin-hashed.bin 41652231" "gsh-2015-host-1.bin0-hashed.bin 68660142" "uk-2006-06-1.bin-hashed.bin 80644903" "uk-union-1.bin-hashed.bin 133633041")
APPS=("0" "2" "1" "4" "5")

for G in "${GRAPHS[@]}"
do
    for A in "${APPS[@]}"
    do
        CMD="./main ${A} ${GRAPHS_PERFIX}/${G} 20"
        echo "Command=${CMD}"
        ${CMD}
    done
done
exit;
