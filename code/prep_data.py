import pandas as pd
import numpy as np
import pickle
import os

# load data and relevant features
all_df = pd.read_csv(csv_data_path)

input_cols = ['GENDER', 'Age at Tx', 'Type Limited Categories', 'SUM OF MM', 'Number of Tx', 'Mode of Dialysis',
				'Class I - AAMM', 'Class II - AAMM', 'Induction', 'CIT', 'ANY Pre Tx', 'Pre Spec C1', 'Pre Spec C2']

label = ['De Novo Ab']

df = all_df[input_cols + label]

# tidy data

# if no test done, assume no Ab present.
df['Pre Spec C1'].replace(np.nan, 0, inplace=True)
df['Pre Spec C2'].replace(np.nan, 0, inplace=True)

df.rename(columns={"GENDER": "Gender",
					"Type Limited Categories": "Donor Type",
					"SUM OF MM": "HLA MM",
					"Class I - AAMM": "AA MM - Class I",
					"Class II - AAMM": "AA MM - Class II",
				   	"PRE TX C1": "ANY PRE TX C1",
				   	"PRE TX C2": "ANY PRE TX C2",
				   	"Pre Spec C1": "Pre Spec C1 - count",
				   	"Pre Spec C2": "Pre Spec C2 - count"}, inplace=True)

no_pre_spec_df = df[df['ANY Pre Tx']==0]  # only get those with no pre spec

# drop Any pre tx, only used to get no pre spec dataset
df.drop(columns=["ANY Pre Tx"], inplace=True)
no_pre_spec_df.drop(columns=["ANY Pre Tx"], inplace=True)

# pickle
with open(os.path.join("data", "cleaned_df_with_missing.pickle"), 'wb') as handle:
	pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join("data", "cleaned_df_with_missing_no_pre_spec.pickle"), 'wb') as handle:
	pickle.dump(no_pre_spec_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
