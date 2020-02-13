import pandas as pd
import numpy as np
from anytree import Node, RenderTree

class C45:
    def __init__(self):
        self.root = Node('DTL')

    def handleMissingValue(self, df):
        columns = df.columns.to_list()
        for col in columns:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

        return df
    
    def handleContinuousValue(self, df, attribute) :
        thresholds = []
        for i in range(len(df[attribute])-1) :
            threshold = (df[attribute][i] + df[attribute][i+1]) / 2
            thresholds.append(threshold)
        return thresholds
