import pandas as pd
import numpy as np

class c45:
    def __init__(self):
        self.root = Node('DTL')

    def handleMissingValue(df):
        columns = df.columns.to_list()
        for col in columns:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

        return df