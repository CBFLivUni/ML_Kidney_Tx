import pandas as pd
import pickle5 as pickle  # pickled with python 3.8+ and this is python3.6, this is a workaround.
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, scale
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import os

# decoded data feat names
with open(os.path.join('pickles', 'pre_decoded_feat_names.pickle'), 'rb') as handle:
	decoded_feat_names = pickle.load(handle)

#rename to match names in text
decoded_feat_names_new = []

for n in decoded_feat_names:
	if n == 'Age at Tx':
		decoded_feat_names_new.append('Age')
	elif n == 'Donor Type':
		decoded_feat_names_new.append('Type of transplant')
	elif n == 'Pre Spec C1 - count':
		decoded_feat_names_new.append('Pre-Tx HSA - Class I')
	elif n == 'Pre Spec C2 - count':
		decoded_feat_names_new.append('Pre-Tx HSA - Class II')
	else:
		decoded_feat_names_new.append(n)

decoded_feat_names = decoded_feat_names_new

# model and data paths.
model_group = [
	{'plot_dir': os.path.join('code', 'figs', 'shap' 'model_1'),
	 'data': os.path.join('data', 'pickle', 'data', 'xgb_train_0_y_all.pickle'),
	 'model': os.path.join('data', 'pickle', 'models', 'xgb_train_0_y_all.pickle'),
	 'clf': 'xgb'},
	{'plot_dir': os.path.join('code', 'figs', 'shap' 'model_2'),
	 'data': os.path.join('data', 'pickle', 'data', 'xgb_train_6_n_all.pickle'),
	 'model': os.path.join('data', 'pickle', 'models', 'xgb_train_6_n_all.pickle'),
	 'clf': 'xgb'},
	{'plot_dir': os.path.join('code', 'figs', 'shap' 'model_3'),
	 'data': os.path.join('data', 'pickle', 'data', 'xgb_train_5_n_all.pickle'),
	 'model': os.path.join('data', 'pickle', 'models', 'xgb_train_5_n_all.pickle'),
	 'clf': 'xgb'},
]

# feature importances for all dataframes.
all_feat_imps = []

for count, group in enumerate(model_group):
	os.chdir(group['plot_dir'])

	# open best model and matching data
	with open(group['data'], 'rb') as handle:
		data = pickle.load(handle)

	with open(group['model'], 'rb') as handle:
		model = pickle.load(handle)

	# if xgboost model
	if group['clf'] == 'xgb':
		model = model.best_estimator_

	# set style
	sns.set_context("paper")

	# Set the font to be serif, rather than sans, and figsize
	sns.set(font='serif', font_scale=2, rc={'figure.figsize': (11, 11)})

	# Make the background white, and specify the
	# specific font family
	sns.set_style("white", {
		"font.family": "serif",
		"font.serif": ["Times", "Palatino", "serif"]
	})

	# xgboost
	xgboost_feat_imps = pd.DataFrame(data={'Features': decoded_feat_names,
										   'Importance': model.feature_importances_})

	xgboost_feat_imps.sort_values(by='Importance', ascending=False, inplace=True)

	ax = sns.barplot(data=xgboost_feat_imps[0:10],
					x='Importance',
					y='Features',
					color='#808080',
					edgecolor='#000000',
					linewidth=1.5,
					zorder=3)
	ax.grid(axis='x', which='major', zorder=0)

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	plt.tight_layout()
	plt.savefig("feature_importance.png")
	plt.close()

	# add to all_feat_imps
	all_feat_imps.append(xgboost_feat_imps)

# calculate mean feature importances for all dfs

all_feat_imps_df = (pd.concat(all_feat_imps)
		    		.groupby('Features')
					.agg(avg=('Importance', 'mean'), std=('Importance', 'std')))

all_feat_imps_df.sort_values(by='avg', ascending=False, inplace=True)

ax = sns.barplot(data=all_feat_imps_df,
					x='avg',
					y=all_feat_imps_df.index,
					color='#808080',
					edgecolor='#000000',
					linewidth=1.5,
					zorder=3)
ax.grid(axis='x', which='major', zorder=0)
_, caplines, _ = ax.errorbar(y=range(len(all_feat_imps_df)),
		    x=all_feat_imps_df['avg'],
			xerr=all_feat_imps_df['std'],
			capsize=10,
			xlolims= True,
			fmt='none',
			ecolor='black',
			zorder=2)
caplines[0].set_marker('|')
plt.xlabel('Importance')

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()
plt.savefig("all_feature_importance.png")
plt.close()
