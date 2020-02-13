from id3 import Id3
from c45 import C45
import pandas as pd
    
def main():    
    id3_iris = Id3()

    # ID3
 
    # iris dataset
    print("Iris Dataset dengan ID3")
    print("---------------------------------------------------------------")
    iris_dataframe = pd.read_csv("iris.csv")
    iris_target = iris_dataframe["species"]
    iris_data_training = iris_dataframe.drop("species", axis=1)
    iris_feature = iris_data_training.columns.to_list()
    id3_iris.fit(iris_data_training, iris_target, iris_feature)
    id3_iris.print_tree()
    print()
    print("---------------------------------------------------------------")
    print()
    
    # C45
    c45_iris = C45()

    # iris dataset
    print("Iris Dataset dengan C45")
    print("---------------------------------------------------------------")
    iris_dataframe_c45 = pd.read_csv('iris.csv')
    iris_target_c45 = iris_dataframe_c45['species']
    iris_data_training_c45 = iris_dataframe_c45.drop('species', axis=1)
    iris_features_c45 = iris_data_training_c45.columns.to_list()
    c45_iris.fit(iris_data_training_c45, iris_target_c45, iris_features_c45)
    c45_iris.print_tree()
    print()
    print("---------------------------------------------------------------")
    print()

if __name__ == "__main__":
    main()
    
