import logging

from utils.file_reader import FileReader
from transformers.drop_column_transformer import *
from transformers.imputation_transformer import *
from splitter_service.splitter_worker import SplitterWorker
from transformers.standard_scaler_transformer import StandardScalerTransformer
from transformers.transformer_service import *


class Pipe:
    def __init__(self, config) -> None:
        self.config = config
        self.graph = config['transformations']['graph']
        self.df = None
        self.X = None
        self.y = None

    def read_input(self):
        input_file = self.config['global_params']['input_file']
        index_col = self.config['global_params']['index_col']

        fr = FileReader(input_file, index_col)

        df = fr.read_file()

        logging.info("File read into dataframe with shape: %s", df.shape)

        self.df = df

    def execute(self):
        # read file into pandas dataframe
        self.read_input()

        # drop unwanted columns
        df_cols_removed = DropColumnTransformer(
            self.df, self.config).drop_columns()

        # cleanup dataframe according to config rules (impute etc.)
        self.df = ImputationTransformer(
            df_cols_removed, self.config).transform()

        # split dataframe to categorical and numerical variables
        splitter = SplitterWorker(df_cols_removed, self.config)
        df_cat, df_num = splitter.split_dataframe()

        # transform numeric data into np.ndarray before transformations and then apply scaling (for now)
        data = TransformerService(df_num).to_struct_array()

        # TODO: encode categorical variables