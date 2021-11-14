import logging


class DropColumnTransformer:
    def __init__(self, dataframe, config):
        self.data = dataframe
        self.config = config

    def drop_columns(self):
        logging.info('** STEP: Dropping unwated columns from dataframe')

        cols = self.config['transformations']['drop_cols']['colnames']

        if len(cols) > 0:
            for col in cols:
                logging.info(
                    '    DROP COLUMN %s removed from dataframe', col)
                self.data.drop(col, axis='columns', inplace=True)

            logging.info('    %s columns dropped from dataframe. New shape is: %s', len(
                cols), {self.data.shape})
        else:
            logging.warning(
                "Nothing to drop, keeping all columns in dataframe")

        logging.info(
            '** COMPLETED STEP: Dropping unwated columns from dataframe')

        return self.data
