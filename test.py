from id3 import Id3
from c45 import C45
import pandas as pd
    
def main():    
    id3 = Id3()
    c45 = C45()

    # dataframe = pd.read_csv("kumpul.csv")
    # target = dataframe["Aktivitas"]
    # data_training = dataframe.drop("Aktivitas", axis=1)
    
    # ID3
    dataframe = pd.read_csv("play_tennis.csv")
    target = dataframe["play"]
    data_training = dataframe.drop("play", axis=1)
    data_training = data_training.drop("day", axis=1)
    

    feature = data_training.columns.to_list()
    id3.fit(data_training, target, feature)
    id3.print_tree()
    
    #C45
    dataframe_c45 = pd.read_csv('test.csv')
    print(c45.handleContinuousValue(dataframe_c45, 'col1'))
    print(c45.handleContinuousValue(dataframe_c45, 'col2'))
    print(c45.handleContinuousValue(dataframe_c45, 'col3'))
    


if __name__ == "__main__":
    main()
    
