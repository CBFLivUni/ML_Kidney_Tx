import matplotlib.pyplot as plt
from dython.nominal import associations
import pickle
import os

# unsensitized group, those who had NO Ab prior
# all group, all p's unsensitized and presensitized
dataset_dict = {
	'unsens': {'f': 'cleaned_df_with_missing_no_pre_spec', 'suffix': '_unsens'},
	'all': {'f': 'cleaned_df_with_missing', 'suffix': '_all'}
	}

# loop over both datasets
for d_set, val in dataset_dict.items():

	with open(os.path.join(val['f'] + ".pickle"), 'rb') as handle:
		df = pickle.load(handle)

	# drop relevant cols
	if d_set == 'unsens':
		df.drop(columns=['Pre Spec C1 - count', 'Pre Spec C2 - count'], inplace=True)

	# bug where if don't tell it to drop feats error, even though no nans
	fig, ax_heat = plt.subplots(1, 1, figsize=(10, 10))
	association = associations(df,
								plot=True,
								nan_strategy='drop_samples',
								ax=ax_heat,
								filename=os.path.join('data', 'figs', 'associations', val['suffix'] + '.png'))
	ax_heat.set_title('Associations')
	ax_heat.margins(0.9, 0.9)

	corr_dna = association['corr']['De Novo Ab']
	corr_dna.drop('De Novo Ab', inplace=True)
 
	# use absolute values to show relationship, so that show highest correlation at top, regardless if negative
	abs_corr_dna = corr_dna.abs()
	sorted_corr_dna = abs_corr_dna.sort_values()  # won't sort without creating a copy

	fig, ax_dna = plt.subplots(1, 1, figsize=(5, 6))
	ax_dna.grid(axis='x', which='major', zorder=0)
	ax_dna.barh(sorted_corr_dna.index, sorted_corr_dna, zorder=3, color='black')
	plt.title("Correlations to De Novo Ab development")
	plt.xlabel("Correlation")
	plt.tight_layout()
	plt.savefig(os.path.join('data', 'figs', 'corr_denovo' + val['suffix'] + '.png'))
	plt.close()
