import pandas as pd
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
from dython.nominal import associations
import statsmodels.api as sm
from scipy.stats import tukey_hsd
from statsmodels.formula.api import ols
import os

# Set the font to be serif, rather than sans
sns.set(font='serif', font_scale=2)

# Make the background white, and specify the specific font family
sns.set_style("white", {
	"font.family": "serif",
	"font.serif": ["Times", "Palatino", "serif"]
})

with open(os.path.join('data', 'pickles', 'cleaned_df_with_missing.pickle'), 'rb') as handle:
	df = pickle.load(handle)

with open(os.path.join('data', 'pickles', 'cleaned_df_with_missing_no_pre_spec.pickle'), 'rb') as handle:
	df_no_pre_spec = pickle.load(handle)

df.rename(columns={'Age at Tx': 'Age',
				   'Donor Type': 'Type of transplant',
				   'Pre Spec C1 - count': 'Pre-Tx HSA - Class I',
				   'Pre Spec C2 - count': 'Pre-Tx HSA - Class II'
				   }, inplace=True)

df_no_pre_spec.rename(columns={'Age at Tx': 'Age',
				   'Donor Type': 'Type of transplant',
				   'Pre Spec C1 - count': 'Pre-Tx HSA - Class I',
				   'Pre Spec C2 - count': 'Pre-Tx HSA - Class II'
				   }, inplace=True)

label_colours = {'Developed': '#C5C6D0', 'Not Developed': '#787276'}
f1_colours = {'with SMOTE': '#C5C6D0', 'without SMOTE': '#787276'}

# clean data
replace = {
	'De Novo Ab': {0: 'Not Developed', 1: 'Developed'},
	'Gender': {0: 'Male', 1: 'Female'},
	'Type of transplant': {0: 'Living', 1: 'Cadaveric'},
	'Mode of Dialysis': {0: 'No', 1: 'Yes'},
	'Induction': {0: 'Non-depleting', 1: 'Depleting'},
	}

df.replace(to_replace=replace, inplace=True)
df_no_pre_spec.replace(to_replace=replace, inplace=True)

df['Pre-Tx HSA - Class I'] = df['Pre-Tx HSA - Class I'].astype(int)
df['Pre-Tx HSA - Class II'] = df['Pre-Tx HSA - Class II'].astype(int)
df_no_pre_spec['Pre-Tx HSA - Class I'] = df_no_pre_spec['Pre-Tx HSA - Class I'].astype(int)
df_no_pre_spec['Pre-Tx HSA - Class II'] = df_no_pre_spec['Pre-Tx HSA - Class II'].astype(int)

plot_dir = os.path.join('code', 'figs', 'uni_plots')

numerical = ['Age', 'HLA MM', 'Number of Tx', 'AA MM - Class I', 'AA MM - Class II',
				'CIT', 'Pre-Tx HSA - Class I', 'Pre-Tx HSA - Class II']
categorical = ['Gender', 'Type of transplant', 'Mode of Dialysis', 'Induction']

for d in [df, df_no_pre_spec]:
	for y in numerical:
		# plot violin and boxplot

		# also do stacked bar for Number of Tx
		if y == "Number of Tx":
			fig, ax = plt.subplots(figsize=(9, 9))

			prop_df = df

			prop_df['De Novo Ab'] = pd.Categorical(prop_df['De Novo Ab'], ['Not Developed','Developed'])
			
			sns.histplot(
				data=prop_df,
				ax=ax,
				x="De Novo Ab",
				hue="Number of Tx",
				multiple="fill",
				stat="proportion",
				palette=sns.color_palette("gray", len(prop_df['Number of Tx'].unique())),
				legend=False,
				discrete=True, shrink=.8
			)

			ax.legend(['4', '3', '2', '1'],
			 		title='Number of Tx',
					loc="best",
					fancybox=False,
					framealpha=1,  # opaque
					facecolor='w',
					edgecolor='0')
			
			# Hide the right and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)

			# Only show ticks on the left and bottom spines
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')

			f = plot_dir + "\\all\\" + "De Novo Ab_" + y + "_stacked_bar.png"
			plt.tight_layout()
			fig.savefig(fname=f)


		for p_type in ['box', 'violin', 'box_no_jitter']:

			if p_type == 'violin':
				fig, ax = plt.subplots(figsize=(9, 9))
				violin_fig = sns.violinplot(data=d,
											x="De Novo Ab",
											y=y,
											inner=None,
											cut=0,
											palette=label_colours,
											order=["Not Developed", "Developed"])

				# jitter
				violin_fig = sns.swarmplot(x='De Novo Ab',
										y=y,
										data=d,
										order=["Not Developed", "Developed"],
										color="#232023")

				fig = violin_fig.get_figure()
				plt.ylabel(y)
				plt.xlabel("De novo HSA")

			elif p_type == 'box':
				# box with jitter
				fig, ax = plt.subplots(figsize=(9, 9))
				box_fig = sns.boxplot(data=d,
										x="De Novo Ab",
										y=y,
										linewidth=1.5,
										palette=label_colours,
										order=["Not Developed", "Developed"],
										showfliers=False)  # hide outliers as jitter shows them

				# jitter
				box_fig = sns.swarmplot(x='De Novo Ab',
								y=y,
								data=d,
								order=["Not Developed", "Developed"],
								color="#232023")

				fig = box_fig.get_figure()
				plt.ylabel(y)
				plt.xlabel("De novo HSA")

			elif p_type == 'box_no_jitter':
				# box without jitter
				fig, ax = plt.subplots(figsize=(9, 9))
				box_fig = sns.boxplot(data=d,
									  x="De Novo Ab",
									  y=y,
									  linewidth=1.5,
									  palette=label_colours,
									  order=["Not Developed", "Developed"],
									  showfliers=True)

				fig = box_fig.get_figure()
				plt.ylabel(y)
				plt.xlabel("De novo HSA")

			# choose dir
			if d["Pre-Tx HSA - Class I"].sum() == 0:
				# then must be no pre spec
				subdir = "\\no_pre_spec\\"
			else:
				subdir = "\\all\\"

			# Hide the right and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)

			# Only show ticks on the left and bottom spines
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('bottom')

			f = plot_dir + subdir + "De Novo Ab_" + y + "_" + p_type + ".png"
			plt.tight_layout()
			fig.savefig(fname=f)

for d in [df, df_no_pre_spec]:
	for y in categorical:
		fig, ax = plt.subplots(figsize=(9, 9))

		prop_df = (d[y]
					.groupby(d["De Novo Ab"])
					.value_counts(normalize=True)
					.rename('Proportion')
					.reset_index())

		bar_fig = sns.barplot(data=prop_df,
						x=y,
						y='Proportion',
						hue="De Novo Ab",
						linewidth=1.5,
						edgecolor='black',
						palette=label_colours,
						)

		plt.legend(loc="best",
					fancybox=False,
					framealpha=1,  # opaque
					facecolor='w',
					edgecolor='0')

		fig = bar_fig.get_figure()

		# Hide the right and top spines
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		# Only show ticks on the left and bottom spines
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')

		# choose dir
		if d["Pre-Tx HSA - Class I"].sum() == 0:
			# then must be no pre spec
			subdir = "\\no_pre_spec\\"
		else:
			subdir = "\\all\\"

		f = plot_dir + subdir + "De Novo Ab_" + y + ".png"
		plt.tight_layout()
		fig.savefig(fname=f)


# f1 results
list_f1_results = [{'dir': os.path.join('data', 'pickles', 'f1_results_all.pickle'),
					'subdir': "\\all\\",
					'name': 'all'},
					{'dir': os.path.join('data', 'pickles', 'f1_results_unsens.pickle'),
					'subdir': "\\no_pre_spec\\",
					'name': 'no_pre_spec'}
]

with open(list_f1_results[0]['dir'], 'rb') as handle:
		all_f1_df = pickle.load(handle)

all_f1_df.sort_values('f1')

for_anova_df = all_f1_df[['model', 'f1', 'SMOTE']]

model = ols('f1 ~ C(model) + C(SMOTE) + C(model):C(SMOTE)',
			data=for_anova_df).fit()

anova_results = sm.stats.anova_lm(model, typ=2)
anova_results.round(5)

# post hoc
res_model = tukey_hsd(for_anova_df[for_anova_df['model'] == 'DT']['f1'],
					  for_anova_df[for_anova_df['model'] == 'RF']['f1'],
					  for_anova_df[for_anova_df['model'] == 'catboost']['f1'],
					  for_anova_df[for_anova_df['model'] == 'xgboost']['f1'])
res_smote = tukey_hsd(for_anova_df[for_anova_df['SMOTE'] == 'y']['f1'],
					  for_anova_df[for_anova_df['SMOTE'] == 'n']['f1'])

print(res_model)
print(res_smote)

with open(list_f1_results[1]['dir'], 'rb') as handle:
	unsens_f1_df = pickle.load(handle)

all_f1_df.groupby(['model', 'SMOTE']).mean()
all_f1_df.groupby(['model', 'SMOTE']).std()

unsens_f1_df['f1'].mean()
unsens_f1_df['f1'].std()

for res in list_f1_results:

	with open(res['dir'], 'rb') as handle:
		f1_df = pickle.load(handle)

	f1_df['model'].replace({'catboost': 'CB',
							'xgboost': 'XGB',
							'DT': 'CART'}, inplace=True)
	f1_df['SMOTE'].replace({'y': 'with SMOTE', 'n': 'without SMOTE'}, inplace=True)

	f1_plot = sns.catplot(data=f1_df,
							kind="bar",
							x="model",
							y="f1",
							hue='SMOTE',
							ci="sd",
							palette=f1_colours,
							height=9,
							aspect=1,
							legend=False,  # easier to do in matplotlib below
							linewidth=1.5,
							edgecolor='black',
							capsize=.15,
							errwidth=1.5,
							errcolor='0'
						)
	f1_plot.despine(left=False)
	f1_plot.set(ylim=(0, None))
	f1_plot.set_axis_labels("Classifier", "F1")
	plt.grid(axis='y', color='black', linewidth=0.5)
	plt.legend(loc="best",
				fancybox=False,
				framealpha=1,  # opaque
				facecolor='w',
				edgecolor='0')
	plt.tight_layout()
	f1_plot.savefig(plot_dir + res['subdir'] + "f1_" + res['name'] + ".png")

# associations
# taken from processing server and remade figs for publication quality.

dataset_dict = {
	'unsens': {'f': 'cleaned_df_with_missing_no_pre_spec', 'suffix': 'no_pre_spec'},
	'all': {'f': 'cleaned_df_with_missing', 'suffix': 'all'}
	}

# loop over both datasets
for d_set, val in dataset_dict.items():

	with open(os.path.join('data', 'pickles', val['f'] + ".pickle"), 'rb') as handle:
		df_association = pickle.load(handle)

	df_association.rename(columns={'Age at Tx': 'Age',
				   'Donor Type': 'Type of transplant',
				   'Pre Spec C1 - count': 'Pre-Tx HSA - Class I',
				   'Pre Spec C2 - count': 'Pre-Tx HSA - Class II'
				   }, inplace=True)

	# drop relevant cols
	if d_set == 'unsens':
		df_association.drop(columns=['Pre-Tx HSA - Class I', 'Pre-Tx HSA - Class II'], inplace=True)

	# bug where if don't tell it to drop feats error, even though no nans
	sns.set(font='serif', font_scale=1.3)
	fig, ax_heat = plt.subplots(1, 1, figsize=(15, 15))
	# adjust plot size so fits on fig
	# https://pythonguides.com/matplotlib-subplots_adjust/
	plt.subplots_adjust(left=0.2,
						bottom=0.11,
						right=0.9,  # default
						top=0.9)  # default
	association = associations(df_association,
								plot=True,
								nan_strategy='drop_samples',
								ax=ax_heat,
								filename=os.path.join('code', 'figs', 'uni plots', val['suffix'] + "\\" + "all_correlation" + '.png'))

	# just for dna
	# seaborn needs setting again because had to change it for heatmap
	sns.set(font='serif', font_scale=2)
	sns.set_style("white", {
		"font.family": "serif",
		"font.serif": ["Times", "Palatino", "serif"]
	})

	corr_dna = association['corr']['De Novo Ab']
	corr_dna.drop('De Novo Ab', inplace=True)
	# use absolute values to show relationship, so that show highest correlation at top, regardless if negative
	abs_corr_dna = corr_dna.abs()
	sorted_corr_dna = abs_corr_dna.sort_values()  # won't let me sort without creating a copy

	fig, ax_dna = plt.subplots(1, 1, figsize=(9, 9))
	plt.grid(axis='x', color='black', linewidth=0.5)
	ax_dna.barh(sorted_corr_dna.index,
				sorted_corr_dna,
				color='#787276',
				linewidth=1.5,
				edgecolor='black'
	)
	plt.xlabel("Correlation")
	plt.tight_layout()

	# Hide the right and top spines
	ax_dna.spines['right'].set_visible(False)
	ax_dna.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax_dna.yaxis.set_ticks_position('left')
	ax_dna.xaxis.set_ticks_position('bottom')

	plt.savefig(os.path.join('code', 'figs', 'uni plots' + "\\" + val['suffix'] + "\\" + "dna_correlation" + '.png'))
	plt.close()


# only living donors

living_df = df[df['Type of transplant'] == "Living"]

for p_type in ['box', 'violin', 'box_no_jitter']:

	if p_type == 'violin':
		fig, ax = plt.subplots(figsize=(9, 9))
		violin_fig = sns.violinplot(data=living_df,
									x="De Novo Ab",
									y='CIT',
									inner=None,
									cut=0,
									palette=label_colours,
									order=["Not Developed", "Developed"])

		# jitter
		violin_fig = sns.swarmplot(x='De Novo Ab',
								   y='CIT',
								   data=living_df,
								   order=["Not Developed", "Developed"],
								   color="#232023")

		fig = violin_fig.get_figure()
		plt.ylabel('CIT')
		plt.xlabel("De novo HSA")

	elif p_type == 'box':
		# box with jitter
		fig, ax = plt.subplots(figsize=(9, 9))
		box_fig = sns.boxplot(data=living_df,
							  x="De Novo Ab",
							  y='CIT',
							  linewidth=1.5,
							  palette=label_colours,
							  order=["Not Developed", "Developed"],
							  showfliers=False)  # hide outliers as jitter shows them

		# jitter
		box_fig = sns.swarmplot(x='De Novo Ab',
								y='CIT',
								data=living_df,
								order=["Not Developed", "Developed"],
								color="#232023")

		fig = box_fig.get_figure()
		plt.ylabel('CIT')
		plt.xlabel("De novo HSA")

	elif p_type == 'box_no_jitter':
		# box without jitter
		fig, ax = plt.subplots(figsize=(9, 9))
		box_fig = sns.boxplot(data=living_df,
							  x="De Novo Ab",
							  y='CIT',
							  linewidth=1.5,
							  palette=label_colours,
							  order=["Not Developed", "Developed"],
							  showfliers=True)

		fig = box_fig.get_figure()
		plt.ylabel('CIT')
		plt.xlabel("De novo HSA")

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	f = plot_dir + "\\all\\" + "living_donor_CIT_all" + "_" + p_type + ".png"
	plt.tight_layout()
	fig.savefig(fname=f)


cadav_df = df[df['Type of transplant'] != "Living"]

for p_type in ['box', 'violin', 'box_no_jitter']:

	if p_type == 'violin':
		fig, ax = plt.subplots(figsize=(9, 9))
		violin_fig = sns.violinplot(data=cadav_df,
									x="De Novo Ab",
									y='CIT',
									inner=None,
									cut=0,
									palette=label_colours,
									order=["Not Developed", "Developed"])

		# jitter
		violin_fig = sns.swarmplot(x='De Novo Ab',
								   y='CIT',
								   data=cadav_df,
								   order=["Not Developed", "Developed"],
								   color="#232023")

		fig = violin_fig.get_figure()
		plt.ylabel('CIT')
		plt.xlabel("De novo HSA")

	elif p_type == 'box':
		# box with jitter
		fig, ax = plt.subplots(figsize=(9, 9))
		box_fig = sns.boxplot(data=cadav_df,
							  x="De Novo Ab",
							  y='CIT',
							  linewidth=1.5,
							  palette=label_colours,
							  order=["Not Developed", "Developed"],
							  showfliers=False)  # hide outliers as jitter shows them

		# jitter
		box_fig = sns.swarmplot(x='De Novo Ab',
								y='CIT',
								data=cadav_df,
								order=["Not Developed", "Developed"],
								color="#232023")

		fig = box_fig.get_figure()
		plt.ylabel('CIT')
		plt.xlabel("De novo HSA")

	elif p_type == 'box_no_jitter':
		# box without jitter
		fig, ax = plt.subplots(figsize=(9, 9))
		box_fig = sns.boxplot(data=cadav_df,
							  x="De Novo Ab",
							  y='CIT',
							  linewidth=1.5,
							  palette=label_colours,
							  order=["Not Developed", "Developed"],
							  showfliers=True)

		fig = box_fig.get_figure()
		plt.ylabel('CIT')
		plt.xlabel("De novo HSA")

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	f = plot_dir + "\\all\\" + "cadav_donor_CIT_all" + "_" + p_type + ".png"
	plt.tight_layout()
	fig.savefig(fname=f)
