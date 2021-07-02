#hehe xd
import numpy as np
import pandas as pd
#import matplotlib
import csv
import talib


"""
Final submission function


"""
def getMyPosition():
  pass

"""
Function which reads the provided csv file and returns a pandas dataframe containing the data
Parameters:
    filename: The name of the file which is read
Returns:
    A pandas dataframe containing the data values from the file, each index represents a stock
"""
def get_data(filename):
    with open(filename, 'r') as f:
        try:
            array = []
            reader = csv.reader(f, delimiter = ',')
            for line in reader:
                array.append(line[0].split())
            df = pd.DataFrame(np.array(array), dtype=np.float64)
            #print(df[0])
        except Exception as e:
            print(e)
            pass
    print(df[0])
    return df


# Conventional main python script setup, also testing
def main():
  get_data('prices250.txt')


if __name__ == "__main__":
  main()

