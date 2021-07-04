#hehe xd
import numpy as np
import pandas as pd
import matplotlib
import csv
import talib as tal

# TESTING

"""
Final submission function

:param df: list | list of all price values of all stocks to date

:return: np.array | 100 integers denoting daily position
"""
def getMyPosition(prcHistSoFar):
    #df = get_data('prices250.txt')
    df = pd.DataFrame(prcHistSoFar).T
    print(df[0].describe())
    sl = gen_stocks(df)
    return

"""
Function used to test implementation of new functons
"""
def getMyPositionTest_Kenzo(prcHistSoFar):
    #df = get_data('prices250.txt')
    df = pd.DataFrame(prcHistSoFar).T
    print(df[0])
    c = get_correl(df)
    print(len(c))
    for i in c:
        print("index1={}, index2={}, correlation={}, length"
              .format(i.index(1), i.index(sorted(i)[-2]), sorted(i)[-2]), len(i))
    print(df[0].describe())
    ma = get_sma(df)
    return

"""
Function that generates a list of stocks from the provided data

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all stocks containing sma and correlation
"""
def gen_stocks(df):
    sl = []
    c = get_correl(df)
    ma = get_sma(df)
    for i in range(df.shape[1]):
        sl.append(Stock(df[i]))
        sl[i].set_correl(c[i])
        sl[i].set_sma(ma[i])
    return sl

"""
Function which determines the correlation between two stocks

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

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
                tempCorr.append(tal.CORREL(df[x], df[y], timeperiod = df.shape[0]).iloc[-1])
        correlation.append(tempCorr)
    return correlation
    #return tal.CORREL(s1, s2, timeperiod = 250)

"""
Function that gets the simple moving average of a set of prices

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all moving average trends
"""
def get_sma(df):
    sma = []
    for i in range(df.shape[1]):
        sma.append(tal.SMA(df[i]))
    return sma


# Conventional main python script setup, also testing
def main():
    pricesFile = "./prices250.txt"
    prcAll = loadPrices(pricesFile)
    #getMyPosition(prcAll)
    getMyPositionTest_Kenzo(prcAll)

if __name__ == "__main__":
  main()


def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T

class Stock():
    def __init__(self, data):
        self._sma = []
        self._correl = []
        self._data = data

    def set_data(self, data):
        self._data = data

    def set_correl(self, correl):
        self._correl = correl

    def set_sma(self, sma):
        self._sma = sma

    def get_data(self):
        return self._data

    def get_correl(self):
        return self._correl

    def get_max_correl_index(self):
        return self._correl.index(sorted(self._correl)[-2])

    def get_sma(self):
        return self._sma
