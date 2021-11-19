class NumCatSplitter:
    def __init__(self, dataframe, config) -> None:
        self.dataframe = dataframe
        self.config = config

    def get_num_cols(self):
        num_cols_list = list(
            self.dataframe.dtypes[self.dataframe.dtypes != 'object'].index.values)

        return num_cols_list

    def get_cat_cols(self):
        cat_cols_list = list(
            self.dataframe.dtypes[self.dataframe.dtypes == 'object'].index.values)
        return cat_cols_list

    def get_cols(self):
        return self.get_num_cols(), self.get_cat_cols()
