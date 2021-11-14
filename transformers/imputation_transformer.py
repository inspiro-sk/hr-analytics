import logging
import logging


class ImputationTransformer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.cols = config['transformations']['transformers']['I']['columns']

    def transform(self):
        logging.info('** STEP: Imputation')

        for col in self.cols:
            col_name = col['name']
            i_strategy = col['strategy']

            logging.info("    IMPUTE COLUMN %s with strategy: %s",
                         col_name, i_strategy)

            if i_strategy == 'mean':
                self.data[col_name].fillna(
                    self.data[col_name].mean(), inplace=True)
            elif i_strategy == 'median':
                self.data[col_name].fillna(
                    self.data[col_name].median(), inplace=True)
            elif i_strategy == 'min':
                self.data[col_name].fillna(
                    self.data[col_name].min(), inplace=True)
            elif i_strategy == 'max':
                self.data[col_name].fillna(
                    self.data[col_name].max(), inplace=True)
            elif i_strategy == 'const':
                imputation_const = col['value']
                self.data[col_name].fillna(imputation_const, inplace=True)

        logging.info("** COMPLETED STEP Imputation")

        return self.data
