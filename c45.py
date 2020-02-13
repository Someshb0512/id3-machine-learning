import pandas as pd
import numpy as np
from anytree import Node, RenderTree
import math

class C45:
    def __init__(self):
        self.root = Node('DTL')

    def handleMissingValue(self, df):
        columns = df.columns.to_list()
        for col in columns:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

        return df
    
    # training_data = training dataframe
    # target_attribute = target dataframe
    # attribute = string nama atribut yang nilainya continuous (real)
    def handleContinuousValue(self, training_data, target_attribute, attribute) :
        td = training_data.assign(target=target_attribute)
        td = td.sort_values(by=attribute)
        thresholds = {}
        for i in range(len(td)-1) :
            if (td['target'][i] != td['target'][i+1]) :
                threshold = (td[attribute][i] + td[attribute][i+1]) / 2
                thresholds[threshold] = [0,0]
        for value in td[attribute] :
            for threshold in thresholds.keys() :
                if (value < threshold) :
                    thresholds[threshold][0] += 1
                else :
                    thresholds[threshold][1] += 1
        return thresholds

    # training_data = training dataframe
    # attribute = string nama atribut yang ingin dicari splitInfonya
    def splitInformation(self, training_data, attribute):
        attribute_val = training_data[attribute]
        unique_values_counts = self.count_unique_values(attribute_val)

        for key in unique_values_counts:
            proportion = unique_values_counts[key] / len(attribute_val) 
            sum += -1 * proportion * math.log2(proportion) 
        return sum

    # training_data = training dataframe
    # target = target dataframe
    # selected attribute = string attribute name
    def gainRatio(self, training_data, target, selected_attribute):
        splitInfo = self.splitInformation(training_data, selected_attribute)
        attr_gain = self.gain(target, training_data[selected_attribute])
    
        return (attr_gain/splitInfo)

    


