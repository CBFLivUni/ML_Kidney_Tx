#!/bin/bash
#SBATCH --job-name kidney_tx
#SBATCH --output kidney_tx.out

module purge
module load python/3.6

source ./venv/bin/activate

python prep_data.py  # preprocessing
python associations_de_novo.py  # heatmaps and associations
python pre_dt_rf_cat_xgb.py  # classifiers
python best_results.py  # print best performing classifiers
