from id3 import Id3
import pandas as pd
    
def main():    
    id3 = Id3()
    
    # dataframe = pd.read_csv("kumpul.csv")
    # target = dataframe["Aktivitas"]
    # data_training = dataframe.drop("Aktivitas", axis=1)
    
    dataframe = pd.read_csv("play_tennis.csv")
    target = dataframe["play"]
    data_training = dataframe.drop("play", axis=1)
    data_training = data_training.drop("day", axis=1)
    

    feature = data_training.columns.to_list()
    print(feature)
    id3.fit(data_training, target, feature)
    id3.printtree()
    

if __name__ == "__main__":
    main()
    
