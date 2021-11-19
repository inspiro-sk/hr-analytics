from sklearn.model_selection import train_test_split
from num_cat_split import NumCatSplitter


class SplitterWorker:
    def __init__(self, data, config) -> None:
        self.dataframe = data
        self.config = config
        self.target = config['global_params']['target_col']
        self.cat_cols = config['columns']['categorical_columns']

    def split_dataframe(self):
        # use num_cat-split to achieve this
        splitter = NumCatSplitter(self.dataframe, self.config)

        if len(self.cat_cols) == 0:
            cat_cols = splitter.get_cat_cols()
        else:
            cat_cols = self.cat_cols

        df_cat = self.dataframe[cat_cols]

        return df_cat, self.dataframe[self.dataframe.columns.difference(cat_cols)]

    def create_target(self):
        target_conf = self.target

        if 'derived' in target_conf:
            target_data = self.dataframe[target_conf['derived']['from']]

            # convert Yes/No column to 1/0
            if target_conf['derived']['as'] == 'category':
                self.dataframe[self.target['name']] = target_data.astype(
                    'category').cat.codes

            # TODO: account for other types and add conversions as needed

            # then drop original column from dataframe; only derived will be used
            self.dataframe.drop(
                target_conf['derived']['from'], axis=1, inplace=True)

        return self.dataframe[self.target['name']]

    def create_train_and_test_sets(self, random_state=1, test_size=0.1):

        target = self.create_target()
        features = self.dataframe[self.dataframe.columns.difference(
            [self.target['name']])]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, random_state=random_state, test_size=test_size)

        return X_train, X_test, y_train, y_test
