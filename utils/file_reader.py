import pandas as pd


class FileReader():
    def __init__(self, file_name, index_col):
        self.file_name = file_name
        self.index_col = index_col

    def read_file(self):
        ext = self.file_name.split('.')[-1]

        if ext == 'csv':
            df = pd.read_csv(self.file_name, index_col=self.index_col)

        return df
