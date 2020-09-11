#!/bin/bash

#SBATCH --job-name mood	# good manners rule
#SBATCH --partition	gpu_big 		# or gpu_devel
#SBATCH --nodes	1		# amount of nodes allocated
#SBATCH --ntasks 16 # amount of nodes allocated
#SBATCH --gres=gpu:1		# number of GPUs to use (Per node!) Max 4 per node
#SBATCH --time=2-00:00:00	# hh:mm:ss, walltime (less requested time -> less time in queue)
#SBATCH --mem=48000

#SBATCH -o /gpfs/data/gpfs0/n.tuluptceva/logs/sbatch/%x-%j-%N.out # STDOUT
#SBATCH -e /gpfs/data/gpfs0/n.tuluptceva/logs/sbatch/%x-%j-%N.err # STDERR


#set -ex
#cd /gpfs/data/gpfs0/n.tuluptceva/source/mood/ad_algos/scripts
##singularity exec --nv --bind /gpfs/data/gpfs0/n.tuluptceva/ mood.sif  bash jupyterlab.sh
#
#singularity exec --nv --bind /gpfs/data/gpfs0/n.tuluptceva/ mood.sif "$@"
#
##singularity exec --writable  --bind /gpfs/gpfs0/n.tuluptceva mood_v bash ./jupyterlab.sh
##singularity shell --writable --bind /gpfs/gpfs0/n.tuluptceva mood_v

set -ex
export PATH="/gpfs/data/gpfs0/n.tuluptceva/miniconda3/bin:$PATH"

#bash << endl
"$@"
#endl