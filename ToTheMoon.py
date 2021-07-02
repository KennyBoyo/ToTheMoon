#hehe xd
import numpy as np
import pandas as pd
#import matplotlib
import csv
import talib

"""
Final submission function
Returns:
    vector of 100 integers denoting daily position
"""
def getMyPosition():
    df = get_data('prices250.txt')
    print(df[0]) #example print used to print all 250 days of data for the first stock
    return

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
        except Exception as e:
            print(e)
            pass
    return df


# Conventional main python script setup, also testing
def main():
    getMyPosition()


if __name__ == "__main__":
  main()

