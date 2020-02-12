from id3 import Id3
import pandas as pd
    
def main():    
    id3 = Id3()
    dataframe = pd.read_csv("play_tennis.csv")
    target = dataframe["play"]
    print(id3.entropy(target))

if __name__ == "__main__":
    main()
    
