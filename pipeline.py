from utils.file_reader import FileReader
from transformers.drop_column_transformer import *
from transformers.imputation_transformer import *
from splitter_service.splitter_worker import *


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

        # create training and testing sets
        splitter = SplitterWorker(df_imputed, self.config)
        X_train, X_test, y_train, y_test = splitter.create_train_and_test_sets(
            random_state=42, test_size=0.2)

        print(X_train.shape, X_test.shape, y_train.shape,
              y_test.shape)  # TODO: remove

        # split numeric and categorical data (before scaling and encoding)
        pass
