import pandas as pd
import numpy as np


class TransformerService:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    def to_struct_array(self):
        # TODO: convert pandas dataframe (numeric columns) to numpy array
        np_arr = self.dataframe.to_numpy()
        return np_arr
