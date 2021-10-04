import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('inputs/general_data.csv')

# drop data with NULL values:
df_nans_dropped = df.dropna()

# transform Yes/No to 1/0
# get pridiction value and drop from dataframe
# also drop columns with no prediction value
df_nans_dropped['AttritionNum'] = df_nans_dropped.Attrition.astype(
    'category').cat.codes

df_clean = df_nans_dropped.drop(['Attrition', 'EmployeeCount', 'Over18',
                                 'StandardHours', 'EmployeeID'], axis=1)

# get numeric and categorical columns
num_cols = list(df_clean.dtypes[df_clean.dtypes != 'object'].index.values)
cat_cols = list(df_clean.dtypes[df_clean.dtypes == 'object'].index.values)

# define numerical and categorical columns/frames for processing
df_nums = df_clean[df_clean[num_cols].columns.difference(['AttritionNum'])]
df_cat = df_clean[cat_cols]

label = df_clean['AttritionNum']

# define train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_nums, label, test_size=0.15, random_state=42)

# scale numerical values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

scaler_dump = scaler.fit(X_train)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler_dump, file)

X_test_scaled = scaler.transform(X_test)

# develop polynomial features (as an example)
poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_dump = poly.fit(X_train_scaled)

with open('poly.pkl', 'wb') as file:
    pickle.dump(poly_dump, file)

np.save('outputs/train_num', X_train_poly)
np.save('outputs/train_labels', y_train)

np.save('outputs/test_num', X_test_poly)
np.save('outputs/test_labels', y_test)
