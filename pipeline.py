from utils.file_reader import FileReader
from utils.sql_reader import SQLReader
from transformers.drop_column_transformer import *
from transformers.imputation_transformer import *
from splitter_service.splitter_worker import *
from splitter_service.num_cat_split import NumCatSplitter


class Pipe:
    def __init__(self, config) -> None:
        self.config = config
        self.df = None

    def execute(self):
        # read file into pandas dataframe
        self.df = FileReader(self.config).read_file()

        # read database table (as alternative to reading file)
        # method will be specified in config.yaml
        # self.df = SQLReader(self.config).query('SELECT * FROM general_data')

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

        # split numeric and categorical data (before scaling and encoding)
        num_cols, cat_cols = NumCatSplitter(
            X_train, config=self.config).get_cols()
        print(num_cols, cat_cols)
