import math
import pandas as pd 
from anytree import Node, RenderTree

class Id3:
    def __init__(self):
        self.root = Node('DTL')
    #     #self.max_depth = max_depth

    # return {value : count, ...}
    def count_unique_values(self, attribute_values):
        unique_values_counts = {}
        for value in attribute_values:
            if value not in unique_values_counts:
                unique_values_counts[value] = 0
            unique_values_counts[value] += 1
        return unique_values_counts

    # Entropy
    def entropy(self, attribute_values):
        unique_values_counts = self.count_unique_values(attribute_values)

        sum = 0
        for key in unique_values_counts:
            sum += -1 * unique_values_counts[key] / len(attribute_values) * math.log2(unique_values_counts[key] / len(attribute_values)) 
        return sum

    # Gain Information
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

    # Best Attribute
    def best_attribute(self, data_training, target):
        gains = {}
        for attribute in data_training:
            attr = data_training[attribute]
            gains[attribute] = self.gain(target, attr)
            print('gain', attribute, gains[attribute])
        key = max(gains, key=gains.get)
        return key, gains[key]

    def printtree(self):
        for pre, fill, node in RenderTree(self.root):
            print("%s%s" % (pre, node.name))

    def fit(self, data_training, target_attribute, attributes, p=Node('DTL')):
        '''
        data_training are the training data. 
        target_attribute is the attribute whose value is to be predicted by the tree. 
        attributes is a list of other attributes that may be tested by the learned decision tree. 
        Returns a decision tree that correctly classiJies the given data_training.
        Algorithm
        Create a Root node for the tree
        If all data_training are positive, 
            Return the single-node tree Root, with label = +
        If all data_training are negative, 
            Return the single-node tree Root, with label = -
        If Attributes is empty, 
            Return the single-node tree Root, with label = most common value of target_attribute in data_training
        Otherwise Begin
            A <- the attribute from Attributes that best* classifies data_training
            The decision attribute for Root <- A
            For each possible value, vi, of A,
                Add a new tree branch below Root, corresponding to the test A = vi
                Let data_training,, be the subset of data_training that have value vi for A
                If data_training,, is empty 
                    Then below this new branch add a leaf node with label = most common value of Target attribute in data_training
                Else below this new branch add the subtree
        fit(data_training,,, Targetattribute, Attributes - ( A ) ) )
        End
        Return Root
        The best attribute is the one with highest information gain,
        '''
        # node = DecisionTreeNode(examples)

        dictionary = self.count_unique_values(target_attribute)
        for key in dictionary:
            if dictionary[key] == len(data_training):
                # node.label = key
                node = Node(key, parent=p)
                return node
                #return key + " "
        
        # if attributes is None: #< minimum allowed per branch:
        #     node.label = most common value in examples
        #     return " "
        # print(data_training)
        best_attr = self.best_attribute(data_training, target_attribute)[0]
        gain = self.best_attribute(data_training, target_attribute)[1]
        node = best_attr + str(gain)
        print('\nbest', best_attr, gain, '\n')
        # print('target',target_attribute)
        
        # node.decision = bestA
        for value in self.count_unique_values(data_training[best_attr]):
            # print('value', value)
            subset = data_training.loc[data_training[best_attr] == value].drop(best_attr, axis=1)
            # print('t>', subset)
            idx = data_training.index[data_training[best_attr] == value].tolist()
            new_target_attribute = target_attribute.loc[idx]
            if len(subset) > 0:
                # new_target_attribute = []
                # for i in range(len(idx)):
                    # print(i)
                    # new_target_attribute.append(target_attribute[i])
                # print('target', new_target_attribute)
                new_attr = [a for a in attributes  if a != best_attr]
                
                par = self.fit(subset, new_target_attribute, new_attr, p)
                print('parent: ', par)
                print('value: ', value)
                node = Node(best_attr + ' ' + value, parent=par)
                self.root = p
                #node += " " + self.fit(subset, new_target_attribute, new_attr)
        
        return node

    
