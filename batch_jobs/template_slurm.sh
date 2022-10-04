#!/bin/bash

#SBATCH --partition=caps
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
##SBATCH --mem 10000
#SBATCH --time=6:00:00
#SBATCH --output=batch_jobs/job.o%j
##export SHELL=bash
##eval `/cvmfs/spt.opensciencegrid.org/py3-v2/setup.sh`
#cd /u/home/s/srinirag/projects_caps/spt3g_software/build
