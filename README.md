# Using Machine Learning to Examine Pre-Transplant Factors Influencing De novo HLA-Specific Antibody Development Post-Kidney Transplant

`code` contains the scripts to run processing

`kidney_tx_analysis.sh` submits the following processing on SLURM:

- `prep_data.py`: basic data cleaning
- `associations_de_novo.py`: generates associations using dython. This is included for completeness, `other_plots.py` is used to generate the publication figures
- `pre_dt_rf_cat_xgb.py`: trains and tests CART, RF, Catboost and XGBoost classifiers
- `best_results.py`: prints the F1 scores of classifier

`shap_top_3_models.py`: run SHAP and generate figures for the top 3 best performing models
`simple_feature_importance.py`: calculate feature importance for the top 3 best performing models and generate figures
`other_plots.py`: generate all other figures, including for exploratory data analysis.
