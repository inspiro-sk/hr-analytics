from utils.file_reader import FileReader
from transformers.drop_column_transformer import *
from transformers.imputation_transformer import *


class Pipe:
    def __init__(self, config) -> None:
        self.config = config
        self.df = None

    def execute(self):
        # read file into pandas dataframe
        self.df = FileReader(self.config).read_file()

        # drop unwanted columns
        df_cols_removed = DropColumnTransformer(
            self.df, self.config).drop_columns()

        # cleanup dataframe according to config rules (impute etc.)
        df_imputed = ImputationTransformer(
            df_cols_removed, self.config).transform()
