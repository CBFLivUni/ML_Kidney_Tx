import pandas as pd
import pickle
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

files = {
	'all': os.path.join("data", "f1_results_all.pickle"),
	'unsens': os.path.join("data", "f1_results_unsens.pickle")
	}

for d_name, d_pickle in files.items():

	with open(d_pickle, 'rb') as handle:
		df = pickle.load(handle)

	sorted_df = df.sort_values(by='f1', ascending=False)

	print("--------BEST F1: " + d_name + "--------")
	print(sorted_df[0:10])
