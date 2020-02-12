import math
# import numpy as np

class Id3:
    # def __init__(self):
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

    def fit(self, data_training, target_attribute, attributes):
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
        return None
        

# class Tree:
    
    
