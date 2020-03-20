from c45 import C45
from id3 import Id3
import pandas as pd
import anytree as at

def main():
    # # ID3 
    id3_tennis = Id3()

    # play-tennis dataset
    print("Play-Tennis Dataset dengan ID3")
    print("---------------------------------------------------------------")
    tennis_dataframe = pd.read_csv("play_tennis.csv")
    tennis_target = tennis_dataframe["play"]
    tennis_data_training = tennis_dataframe.drop("play", axis=1)
    tennis_data_training = tennis_data_training.drop("day", axis=1)
    tennis_feature = tennis_data_training.columns.to_list()
    id3_tennis.fit(tennis_data_training, tennis_target, tennis_feature)
    # id3_tennis.print_tree()
    # print(at.RenderTree(id3_tennis.root))
    print(id3_tennis.predict(tennis_dataframe))
    # [node.name for node in at.PreOrderIter(id3_tennis.root)]
    print()
    print("---------------------------------------------------------------")
    print()

# ------------------------------------------------------------- #

    # C45
    # c45_iris = C45()

    # # iris dataset
    # print("Iris Dataset dengan C45")
    # print("---------------------------------------------------------------")
    # iris_dataframe_c45 = pd.read_csv('iris.csv')
    # iris_target_c45 = iris_dataframe_c45['species']
    # iris_data_training_c45 = iris_dataframe_c45.drop('species', axis=1)
    # iris_features_c45 = iris_data_training_c45.columns.to_list()
    # c45_iris.fit(iris_data_training_c45, iris_target_c45, iris_features_c45)
    # c45_iris.print_tree()
    # print()
    # print("---------------------------------------------------------------")
    # print()

if __name__ == "__main__":
    main()
    
