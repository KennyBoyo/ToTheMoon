#hehe xd
import numpy as np
import pandas as pd
import matplotlib
import csv
import talib as tal

"""
Final submission function

:return: np.array | 100 integers denoting daily position
"""
def getMyPosition():
    df = get_data('prices250.txt')
    c = get_correl(df)
    return

"""
Function used to test implementation of new functons
"""
def getMyPositionTest_Kenzo():
    df = get_data('prices250.txt')
    print(df[0]) #example print used to print all 250 days of data for the first stock
    c = get_correl(df)
    print(len(c))
    for i in c:
        print("index1={}, index2={}, correlation={}, length"
              .format(i.index(1), i.index(sorted(i)[-2]), sorted(i)[-2]), len(i))
    print(df[0].describe())
    return


"""
Function which reads the provided csv file and returns a pandas dataframe containing the data

:param filename: str | The name of the file which is read

:return: pd.Dataframe| dataframe of values from the file, each index represents a stock
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


"""
Function which determines the correlation between two stocks

:param df: pd.Dataframe | Dataframe values 

:return: 2d array | 100x100 array containing correlation values between the stocks at different indices
"""
def get_correl(df):
    correlation = []
    for x in range(df.shape[1]):
        tempCorr = []
        for y in range(df.shape[1]):
            #print("{}, {}, {}".format(correlation, x, y))
            if y < x:
                tempCorr.append(correlation[y][x])
            elif y == x:
                tempCorr.append(1)
            else:
                tempCorr.append(tal.CORREL(df[x], df[y], timeperiod = 250).iloc[-1]) #use df.shape[1] for timeperiod?
        correlation.append(tempCorr)
    return correlation
    #return tal.CORREL(s1, s2, timeperiod = 250)

# Conventional main python script setup, also testing
def main():
    getMyPositionTest_Kenzo()

if __name__ == "__main__":
  main()

