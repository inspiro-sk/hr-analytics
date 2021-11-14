from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split


class SplitterWorker:
    def __init__(self, data, config) -> None:
        self.dataframe = data
        self.target = config['global_params']['target_col']
        self.cat_cols = config['columns']['categorical_columns']

    def split_dataframe(self):
        df = self.dataframe
        cat_cols = self.cat_cols

        if len(cat_cols) == 0:
            cat_cols = list(df.dtypes[df.dtypes == 'object'].index.values)

        df_cat = df[cat_cols]

        return df_cat, self.dataframe[self.dataframe.columns.difference(cat_cols)]

    def create_train_and_test_sets(self, random_state=1, test_size=0.1):
        features = 0
        target = 0

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, random_state=random_state, test_size=test_size)

        return X_train, X_test, y_train, y_test
