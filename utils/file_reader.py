import logging
import os
import pandas as pd


class FileReader():
    def __init__(self, config):
        self.config = config
        self.project_dir = self.config['global_params']['project_dir']
        self.input_file = self.config['global_params']['input_file']
        self.index_col = self.config['global_params']['index_col']

    def read_file(self):
        logging.info("** STEP: Importing source data into dataframe")

        ext = self.input_file.split('.')[-1]

        if ext == 'csv':
            try:
                df = pd.read_csv(os.path.join(
                    self.project_dir, self.input_file), index_col=self.index_col)
            except:
                message = '    Source file specified in config not found'

                logging.error(message)
                raise Exception(message)
        else:
            message = '    Cannot read source file.'
            logging.error(message)
            raise Exception(message)

        logging.info("    File read into dataframe with shape: %s", df.shape)
        logging.info("** COMPLETED STEP: Importing source data into dataframe")

        return df
