#!/bin/bash
#SBATCH -J llm-attacks
#SBATCH -A sail
#SBATCH -p sail
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00 

INITIALTIME=$(date)

#SBATCH -o gcg_individual$INITIALTIME.out
#SBATCH -e gcg_individual$INITIALTIME.err

module load gcc/9.2 cmake python3/3.10 cuda/11.7

source ../../env/bin/activate

INITIALTIME=$(date)
echo "The initial time is " $INITIALTIME
bash run_gcg_individual.sh vicuna behaviors
FINALTIME=$(date)
echo "Finished at " $FINALTIME
echo "Elapsed time is " $FINALTIME-$INITIALTIME
