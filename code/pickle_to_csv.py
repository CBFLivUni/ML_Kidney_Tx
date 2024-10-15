import pickle
import os

with open(os.path.join('data', 'pickles', 'cleaned_df_with_missing.pickle'), 'rb') as handle:
	df = pickle.load(handle)

df.to_csv(os.path.join('data', 'pickles', 'cleaned_df_with_missing.csv'))
