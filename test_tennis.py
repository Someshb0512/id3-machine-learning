from id3 import Id3
from c45 import C45
import pandas as pd
    
def main():    

    # ID3 
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
    id3_tennis.print_tree()
    print()
    print("---------------------------------------------------------------")
    print()

    # C45
    
    c45_tennis = C45()

    # play-tennis dataset
    print("Play-Tennnis Dataset dengan C45")
    print("---------------------------------------------------------------")
    tennis_dataframe_c45 = pd.read_csv('play_tennis.csv')
    tennis_target_c45 = tennis_dataframe_c45['play']
    tennis_data_training_c45 = tennis_dataframe_c45.drop('play', axis=1)
    tennis_data_training_c45 = tennis_data_training_c45.drop('day', axis=1)
    tennis_features_c45 = tennis_data_training_c45.columns.to_list()
    c45_tennis.fit(tennis_data_training_c45, tennis_target_c45, tennis_features_c45)
    c45_tennis.print_tree()
    print()
    print("---------------------------------------------------------------")
    


if __name__ == "__main__":
    main()
    
