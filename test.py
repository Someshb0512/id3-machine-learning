from id3 import Id3
import pandas as pd
    
def main():    
    id3 = Id3()
    dataframe = pd.read_csv("play_tennis.csv")
    target = dataframe["play"]
    data_training = dataframe.drop("day", axis=1)
    data_training = data_training.drop("play", axis=1)
    feature = data_training.columns.to_list()
    print(feature)
    humidity = dataframe["humidity"]
    print(id3.entropy(target))
    print(id3.best_attribute(data_training, target))
    print(id3.fit(data_training, target, feature))
    

if __name__ == "__main__":
    main()
    
