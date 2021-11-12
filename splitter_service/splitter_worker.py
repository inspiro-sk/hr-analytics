class SplitterWorker:
    def __init__(self, data, config) -> None:
        self.dataframe = data
        self.cat_cols = config['columns']['categorical_columns']

    def split_dataframe(self):
        df = self.dataframe
        cat_cols = self.cat_cols

        if len(cat_cols) == 0:
            cat_cols = list(df.dtypes[df.dtypes == 'object'].index.values)

        df_cat = df[cat_cols]

        return df_cat, self.dataframe[self.dataframe.columns.difference(cat_cols)]
