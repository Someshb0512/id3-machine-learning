import pandas as pd
import numpy as np
from anytree import Node, RenderTree
import math
from id3 import Id3

class C45:
    def __init__(self):
        self.root = Node('DTL')

    def count_unique_values(self, attribute_values):
        unique_values_counts = {}
        for value in attribute_values:
            if value not in unique_values_counts:
                unique_values_counts[value] = 0
            unique_values_counts[value] += 1
        return unique_values_counts

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

        sum = 0
        for key in unique_values_counts:
            proportion = unique_values_counts[key] / len(attribute_val) 
            sum += -1 * proportion * math.log2(proportion) 
        return sum

    def entropy(self, attribute_values):
        unique_values_counts = self.count_unique_values(attribute_values)
        sum = 0
        for key in unique_values_counts:
            proportion = unique_values_counts[key] / len(attribute_values) 
            sum += -1 * proportion * math.log2(proportion) 
        return sum

    def gain(self, target, attribute):
        unique_vals_attr = self.count_unique_values(attribute)
        sum = 0
        index = attribute.index.tolist()
        for value in unique_vals_attr:
            values = []
            for i in index: 
                if attribute[i] == value:
                    values.append(target[i])
            sum += unique_vals_attr[value] / len(attribute) * self.entropy(values) 
        return self.entropy(target) - sum

    # training_data = training dataframe
    # target = target dataframe
    # selected attribute = string attribute name
    def gainRatio(self, training_data, target, selected_attribute):
        splitInfo = self.splitInformation(training_data, selected_attribute)
        attr_gain = self.gain(target, training_data[selected_attribute])
    
        return (attr_gain/splitInfo)

    def best_attribute(self, training_data, target):
        gains = {}
        for attribute in training_data:
            attr = training_data[attribute]
            gains[attribute] = self.gainRatio(training_data, target, attribute)
        key = max(gains, key=gains.get)
        return key, gains[key]

    def print_tree(self):
        for pre, fill, node in RenderTree(self.root):
            print("%s%s" % (pre, node.name))

    def fit(self, data_training, target_attribute, attributes, p=Node('DT')):
        dictionary = self.count_unique_values(target_attribute)
        for key in dictionary:
            if dictionary[key] == len(data_training):
                node = Node(key, parent=p)
                return node
        
        best_attr = self.best_attribute(data_training, target_attribute)[0]
        gain_ratio = self.best_attribute(data_training, target_attribute)[1]
        node = Node(best_attr + " " + str(gain_ratio), parent=p)
        
        for value in self.count_unique_values(data_training[best_attr]):
            subset = data_training.loc[data_training[best_attr] == value].drop(best_attr, axis=1)
            idx = data_training.index[data_training[best_attr] == value].tolist()
            new_target_attribute = target_attribute.loc[idx]
            
            if len(subset) > 0:
                new_attr = [a for a in attributes  if a != best_attr]
                node2 = Node(best_attr + ' -> ' + str(value), parent=node)
                par = self.fit(subset, new_target_attribute, new_attr, node2)
        
        self.root = p
        return node


