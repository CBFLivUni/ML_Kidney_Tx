import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import pickle
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE, SMOTENC
import seaborn as sns
import xgboost as xgb
import os

JOBS = 35

# unsensitized group, those who had NO Ab prior
# all group, all p's unsensitized and presensitized
dataset_dict = {
	'unsens': {'f': 'cleaned_df_with_missing_no_pre_spec', 'suffix': '_unsens'},
	'all': {'f': 'cleaned_df_with_missing', 'suffix': '_all'}
	}

# loop over both datasets
for d_set, val in dataset_dict.items():

	results_df = pd.DataFrame()

	for smote in ['y', 'n']:

		with open(os.path.join('data', val['f'] + '.pickle'), 'rb') as handle:
			df = pickle.load(handle)
		
		# reset idx otherwise unsens cohort have non consecutive idx's which causes problems when splitting.
		df.reset_index(drop=True, inplace=True)

		# if unsens, then drop pre spec cols as all 0
		if d_set == 'unsens':
			input_df = df.drop(columns=['De Novo Ab', 'Pre Spec C1 - count', 'Pre Spec C2 - count'])
		else:
			input_df = df.drop(columns=['De Novo Ab'])

		output = df['De Novo Ab']

		# encode any data that requires it
		ord_feats = ['Gender','Donor Type', 'Induction', 'Mode of Dialysis']

		idx_cat_feats = [list(input_df.columns).index(feat) for feat in ord_feats]

		# all input features in order of being encoded
		feats_enc_order = ord_feats + list(input_df.drop(columns=ord_feats).columns)

		ord_enc = OrdinalEncoder()
		ord_enc_vals = ord_enc.fit_transform(input_df[ord_feats])

		# concat encoded inputs + all other inputs
		transformed_inputs = np.hstack((ord_enc_vals, input_df.drop(columns=ord_feats).values))

		# easier to use dict for encoding outputs as labels are simple
		transformed_outputs = output.replace({"Y": 1, "N": 0})

		# generate features names for encoded features
		inverse_feature_names = []

		# get ordinal features
		for feature in ord_feats:
			inverse_feature_names.append(feature)

		# get rest of features
		for feature in list(input_df.drop(columns=ord_feats).columns):
			inverse_feature_names.append(feature)

		with open(os.path.join('data', 'pre_decoded_feat_names.pickle'), 'wb') as handle:
			pickle.dump(inverse_feature_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# run 10 k fold CV on DT, RF, catboost and xgboost
		# run RF and DT first
		skf = StratifiedKFold(n_splits=10, shuffle=False)

		# get indexes of test and train data for k_fold cv
		models_data_idx = 10 * []

		for train_index, test_index in skf.split(transformed_inputs, transformed_outputs):
			models_data_idx.append({'train_index': train_index, 'test_index': test_index})

		split = 0  # count splits in data for results_df

		cat_features = ['Gender', 'Donor Type', 'Induction']
		idx_cat_feats = [list(input_df.columns).index(feat) for feat in cat_features]

		# train all models
		for idx in models_data_idx:
			X_train, X_test = transformed_inputs[idx['train_index']], transformed_inputs[idx['test_index']]
			y_train, y_test = transformed_outputs[idx['train_index']], transformed_outputs[idx['test_index']]

			if smote == 'y':
				# upsample minority class TRAINING DATA ONLY
				X_train, y_train = SMOTENC(categorical_features=idx_cat_feats, random_state=0).fit_resample(X_train, y_train)
				X_train = X_train.round(0)  # round synthentic values, which don't exist in real data

			# train models
			clf_dt = DecisionTreeClassifier(class_weight='balanced')
			clf_dt.fit(X_train, y_train)
			y_pred = clf_dt.predict(X_test)
			dt_f1 = f1_score(y_test, y_pred)

			clf_rf = RandomForestClassifier(class_weight='balanced', random_state=0)
			clf_rf.fit(X_train, y_train)
			y_pred = clf_rf.predict(X_test)
			rf_f1 = f1_score(y_test, y_pred)

			# add results to df
			d = {'f1': [dt_f1, rf_f1], 'model': ['DT', 'RF'], 'split': [split, split],
					'SMOTE': [smote, smote], 'best_params': ['NA', 'NA']}
			results_df = pd.concat([results_df, pd.DataFrame(data=d)])

			save_name_to_model = {'dt': clf_dt, 'rf': clf_rf}

			# save sklearn train data
			with open(os.path.join('data', 'models', 'sklearn_train_' + str(split) + "_" + smote + val['suffix'] + '.pickle'), 'wb') as handle:
				pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

			# save model
			for model in save_name_to_model.keys():
				with open(os.path.join('data', 'models', str(model) + "_" + str(split) + "_" + smote + val['suffix'] + '.pickle'), 'wb') as handle:
					pickle.dump(save_name_to_model[model], handle, protocol=pickle.HIGHEST_PROTOCOL)

			split += 1  # increment split counter
			print('DT- RF: splits complete: ' + str(split) + " / " + str(10))

		# then boosting, because need separate validation set for boosting.
		cat_features = ['Gender', 'Donor Type', 'Induction']
		idx_cat_feats = [list(input_df.columns).index(feat) for feat in cat_features]

		# get indexes of test and train data for k_fold cv
		models_data_idx = 10 * []
		skf = StratifiedKFold(n_splits=10, shuffle=False)
		for rest_index, val_index in skf.split(input_df, output):
			# split test and validation set
			train_index = rest_index[len(val_index):]
			test_index = rest_index[:len(val_index)]
			models_data_idx.append({'train_index': train_index, 'test_index': test_index, 'val_index': val_index})

		split = 0  # count splits in data for results_df

		for idx in models_data_idx:
			# catboost can take the df without OHE
			X_train, X_test, X_val = input_df.iloc[idx['train_index']], input_df.iloc[idx['test_index']], input_df.iloc[idx['val_index']]
			y_train, y_test, y_val = output.iloc[idx['train_index']], output.iloc[idx['test_index']], output.iloc[idx['val_index']]

			if smote == 'y':
				# upsample minority class TRAINING DATA ONLY
				X_train, y_train = SMOTENC(categorical_features=idx_cat_feats, random_state=0).fit_resample(X_train, y_train)
				X_train = X_train.round(0)  # round synthentic values, which don't exist in real data

			train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features, feature_names=list(X_train.columns))
			eval_pool = Pool(data=X_val, label=y_val, cat_features=cat_features, feature_names=list(X_val.columns))

			cat_model = CatBoostClassifier(cat_features=cat_features,
										verbose=50,
										auto_class_weights='Balanced',
										loss_function='Logloss',
										iterations=None,
										learning_rate=None,
										random_state=0)

			grid = {'learning_rate': [x / 1000 for x in range(1, 100)],
					'depth': [6, 8, 10, 12, 16],  # set
					'l2_leaf_reg': [1, 3, 5, 7, 9]}  # set
		
			grid_search_result = cat_model.randomized_search(grid, X=train_pool)

			# train the model
			cat_model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=10, use_best_model=True)
			y_pred = cat_model.predict(X_test)
			cat_f1 = f1_score(y_test.values, y_pred, pos_label=1)

			# save catboost train data, but can't pickle Pool object, so need to just get the data required for now
			for_pool = [X_train, y_train, cat_features]
			with open(os.path.join('data', 'cat_train_pool_' + str(split) + "_" + smote + val['suffix'] + '.pickle'), 'wb') as handle:
				pickle.dump(for_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)

			# XGBoost needs encoded data as inputs
			X_train, X_test, X_val = transformed_inputs[idx['train_index']], transformed_inputs[idx['test_index']], \
										transformed_inputs[idx['val_index']]
			y_train, y_test, y_val = transformed_outputs[idx['train_index']], transformed_outputs[idx['test_index']], \
										transformed_outputs[idx['val_index']]

			# train model
			xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=JOBS, random_state=0)

			param_grid_xgb = {'max_depth': list(range(1, 14)),  # set
								'learning_rate': [x / 1000 for x in range(1, 100)],
								'gamma': [x / 1000 for x in range(1, 100)],  # create range 0.001, 1
								'reg_lambda': list(range(0, 100)),
								'scale_pos_weight': list(range(3, 9))}  # set

			clf_xgb = RandomizedSearchCV(xgb_model,
									param_grid_xgb,
									n_iter=10,
									verbose=1,
									scoring='f1',
									n_jobs=JOBS)

			clf_xgb.fit(X_train, y_train, verbose=50, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
			y_pred = clf_xgb.predict(X_test)
			xgb_f1 = f1_score(y_test, y_pred)

			# add results to df.
			# dict params want to check for catboost
			cat_best_params = {'learning_rate': cat_model.get_params()['learning_rate'],
								'depth': cat_model.get_params()['depth'],
								'l2_leaf_reg': cat_model.get_params()['l2_leaf_reg']}

			d = {'f1': [cat_f1, xgb_f1], 'model': ['catboost', 'xgboost'], 'split': [split, split], 'SMOTE': [smote, smote],
				'best_params': [cat_best_params, clf_xgb.best_params_]}
			results_df = pd.concat([results_df, pd.DataFrame(data=d)])

			save_name_to_model = {'cat': cat_model, 'xgb': clf_xgb}

			# save xgb train data
			with open(os.path.join('data', "xgb_train_" + str(split) + "_" + smote + val['suffix'] + '.pickle'), 'wb') as handle:
				pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

			# save model
			for model in save_name_to_model.keys():
				with open(os.path.join('data', 'models', str(model) + "_" + str(split) + "_" + smote + val['suffix'] + '.pickle'), 'wb') as handle:
					pickle.dump(save_name_to_model[model], handle, protocol=pickle.HIGHEST_PROTOCOL)

			split += 1  # increment split counter
			print('Boosting: splits complete: ' + str(split) + " / " + str(10))

	with open(os.path.join('data', 'pickles', 'f1_results' + val['suffix'] + ".pickle"), 'wb') as handle:
		pickle.dump(results_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


	# grouped barchart of f1
	sns.set_theme(style="whitegrid")
	g = sns.catplot(data=results_df, kind="bar", x="model", y="f1", hue='SMOTE', ci="sd", palette="Set2")
	g.despine(left=True)
	g.set_axis_labels("", "F1")
	g.savefig(os.path.join('code', 'figs', 'RT_DF_catboost_xgboost_f1' + val['suffix'] + '.png'))
