#!/bin/bash
# Sample batchscript to run a parallel python job on HPC using 24 CPU cores

#SBATCH --partition=bch-largemem	 			# queue to be used
#SBATCH --time=00:01:00	 			# Running time (in hours-minutes-seconds)
#SBATCH --job-name=test-compute 			# Job name
#SBATCH --mail-type=BEGIN,END,FAIL 		# send an email when the job begins, ends or fails
#SBATCH --mail-user=david.yarmolinsky@childrens.harvard.edu 	# Email address to send the job status
#SBATCH --output=output_%j.txt 			# Name of the output file
#SBATCH --nodes=1				# Number of compute nodes
#SBATCH --ntasks=24				# Number of cpu cores on one node
#SBATCH --mem=20G

module load anaconda3
python  "/home/ch184656/YarmoPain GUI/runDeposit.py" $SLURM_NTASKS
