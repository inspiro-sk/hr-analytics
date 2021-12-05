import joblib
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

dir_path = os.path.dirname(os.path.realpath(__file__))


df = pd.read_csv(os.path.join(dir_path, 'inputs/general_data.csv'))

# drop data with NULL values:
df.dropna(inplace=True)

# transform Yes/No to 1/0
# get pridiction value and drop from dataframe
# also drop columns with no prediction value
df['AttritionNum'] = df.Attrition.astype(
    'category').cat.codes

df.drop(['Attrition', 'EmployeeCount', 'Over18',
         'StandardHours', 'EmployeeID'], axis=1, inplace=True)

# get numeric and categorical columns
num_cols = list(df.dtypes[df.dtypes != 'object'].index.values)
cat_cols = list(df.dtypes[df.dtypes == 'object'].index.values)

# define numerical and categorical columns/frames for processing
df_nums = df[df[num_cols].columns.difference(['AttritionNum'])]
df_cat = df[cat_cols]

label = df['AttritionNum']

# define train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_nums, label, test_size=0.15, random_state=42)

np.save(os.path.join(dir_path, 'outputs/train_num.npy'), X_train)
np.save(os.path.join(dir_path, 'outputs/train_labels.npy'), y_train)
np.save(os.path.join(dir_path, 'outputs/test_num.npy'), X_test)
np.save(os.path.join(dir_path, 'outputs/test_labels.npy'), y_test)

# scale numerical values
scaler = StandardScaler()

# develop polynomial features (as an example)
poly = PolynomialFeatures(degree=3)

pipe = Pipeline([
    ('scaler', scaler),
    ('poly', poly)
])

pipe_fit = pipe.fit(X_train, y_train)

joblib.dump(pipe_fit, 'pipe_fit.joblib')
