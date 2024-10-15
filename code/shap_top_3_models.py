import os
import pdb

import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

with open(os.path.join('pickles', 'pre_decoded_feat_names.pickle'), 'rb') as handle:
	decoded_feats = pickle.load(handle)

#rename to match names in text
decoded_feat_names_new = []

for n in decoded_feats:
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

decoded_feats = decoded_feat_names_new

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

for group in model_group:
	os.chdir(group['plot_dir'])

	# open best model and matching data
	with open(group['data'], 'rb') as handle:
		data = pickle.load(handle)

	with open(group['model'], 'rb') as handle:
		model = pickle.load(handle)

	# if xgboost model
	if group['clf'] == 'xgb':
		model = model.best_estimator_

	X_train = pd.DataFrame(data=data)

	X_train.columns = decoded_feats

	# remove SMOTE data, i.e. anything that isn't in original dataset.
	# format original dataset like X_train
	df = pd.read_csv(os.path.join('data', 'pickles', 'cleaned_df_with_missing.csv'), index_col=0)

	# reset idx otherwise unsens cohort have non consecutive idx's which causes problems when splitting.
	df.reset_index(drop=True, inplace=True)

	input_df = df.drop(columns=['De Novo Ab'])
	# encode any data that requires it
	ord_feats = ['Gender', 'Type of transplant', 'Induction', 'Mode of Dialysis']

	input_df.rename(columns={'Age at Tx': 'Age',
				   'Donor Type': 'Type of transplant',
				   'Pre Spec C1 - count': 'Pre-Tx HSA - Class I',
				   'Pre Spec C2 - count': 'Pre-Tx HSA - Class II'
				   }, inplace=True)

	idx_cat_feats = [list(input_df.columns).index(feat) for feat in ord_feats]

	# all input features in order of being encoded
	feats_enc_order = ord_feats + list(input_df.drop(columns=ord_feats).columns)

	ord_enc = OrdinalEncoder()
	ord_enc_vals = ord_enc.fit_transform(input_df[ord_feats])

	# concat encoded inputs + all other inputs
	transformed_inputs = np.hstack((ord_enc_vals, input_df.drop(columns=ord_feats).values))

	original_df = pd.DataFrame(data=transformed_inputs)

	original_df.columns = decoded_feats

	# keep only samples which are in both dataframes ie. REMOVE SMOTE SAMPLES
	X_train_no_smote = X_train.merge(original_df, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'both'].drop(columns=["_merge"])

	# reassign that var for ease
	X_train = X_train_no_smote

	style.use('seaborn-dark')

	fig_h = 9
	fig_w = 9

	explainer = shap.TreeExplainer(model)

	shap_values = explainer.shap_values(X_train)

	if group['clf'] == 'rf':
		shap.summary_plot(shap_values[1], X_train, show=False, plot_type='dot')
	else:
		shap.summary_plot(shap_values, X_train, show=False, plot_type='dot')
	f = plt.gcf()
	plt.tight_layout()
	plt.savefig("summary_plot_shap.png")

	plt.figure()  # new fig
	if group['clf'] == 'rf':
		shap.summary_plot(shap_values[1], X_train, plot_type="bar", show=False)
	else:
		shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
	f = plt.gcf()
	plt.tight_layout()
	plt.savefig("bar_summary_plot_shap.png")

	# dependence plot for each feature
	for name in X_train.columns:
		plt.figure()
		if group['clf'] == 'rf':
			shap.dependence_plot(name, shap_values[1], X_train, show=False)
		else:
			shap.dependence_plot(name, shap_values, X_train, show=False)

		f = plt.gcf()
		if name == "Number of Tx":
			a = plt.gca()
			a.set_xticks([1, 2, 3, 4])
		f.set_figheight(fig_h)
		f.set_figwidth(fig_w)
		plt.tight_layout()
		plt.savefig("dependence" + name + ".png")

	# show=false doesn't work
	plt.close('all')

	# dependence plot for pre spec c1 and c2 with each other feature
	for c in ['Pre-Tx HSA - Class I', 'Pre-Tx HSA - Class II']:
		for name in X_train.columns:
			plt.figure()
			if group['clf'] == 'rf':
				shap.dependence_plot(c, shap_values[1], X_train, interaction_index=name, show=False)
			else:
				shap.dependence_plot(c, shap_values, X_train, interaction_index=name, show=False)

			f = plt.gcf()
			f.set_figheight(fig_h)
			f.set_figwidth(fig_w)
			plt.tight_layout()
			plt.savefig("dependence_" + c + name + ".png")

	# decision plot
	plt.figure()
	if group['clf'] == 'rf':
		shap.decision_plot(explainer.expected_value[1], shap_values[1], decoded_feats, feature_order='hclust',
						   link='logit')
	else:
		shap.decision_plot(explainer.expected_value, shap_values, decoded_feats, feature_order='hclust', link='logit')
	f = plt.gcf()
	f.set_figheight(fig_h)
	f.set_figwidth(fig_w)
	plt.tight_layout()
	plt.savefig("decision_plot.png")

	# interaction values
	if group['clf'] == 'rf':
		shap.summary_plot(explainer.shap_interaction_values(X_train)[1], X_train)
	else:
		shap.summary_plot(explainer.shap_interaction_values(X_train), X_train)
	f = plt.gcf()
	f.set_figheight(fig_h)
	f.set_figwidth(fig_w)
	plt.tight_layout()
	plt.savefig("interaction_plot.png")

	plt.close('all')
	
	# living donors
	# 0 is living, 1 is cadaveric have compared against original data
	living_df = X_train[X_train['Type of transplant'] == 0]
	living_shap = shap_values[X_train['Type of transplant'] == 0]

	shap.dependence_plot('CIT', living_shap, living_df, show=False)

	f = plt.gcf()
	f.set_figheight(fig_h)
	f.set_figwidth(fig_w)
	plt.tight_layout()
	plt.savefig("dependence_living_CIT.png")

	cadav_df = X_train[X_train['Type of transplant'] == 1]
	cadav_shap = shap_values[X_train['Type of transplant'] == 1]
	shap.dependence_plot('CIT', cadav_shap, cadav_df, show=False)

	f = plt.gcf()
	f.set_figheight(fig_h)
	f.set_figwidth(fig_w)
	plt.tight_layout()
	plt.savefig("dependence_cadav_CIT.png")
