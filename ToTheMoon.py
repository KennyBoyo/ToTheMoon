#hehe xd
import numpy as np
import pandas as pd
import matplotlib
import csv
import talib as tal


# HI

"""
Class which holds all stock data for a single stock
"""
class Stock():
    def __init__(self, data):
        self._wma = []
        self._correl = []
        self._data = data

    def set_data(self, data):
        self._data = data

    def set_correl(self, correl):
        self._correl = correl

    def set_wma(self, wma):
        self._wma = wma

    def get_data(self):
        return self._data

    def get_correl(self):
        return self._correl

    def get_max_correl_index(self):
        return self._correl.index(sorted(self._correl)[-2])

    def get_wma(self):
        return self._wma

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T

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
    #for s in sl:
        #print(s.get_max_correl_index())
        #print(s.get_wma())
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
    wma = get_wma(df)
    return

"""
Function that generates a list of stocks from the provided data

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all stocks containing wma and correlation
"""
def gen_stocks(df):
    sl = []
    c = get_correl(df)
    wma = get_wma(df)
    for i in range(df.shape[1]):
        sl.append(Stock(df[i]))
        sl[i].set_correl(c[i])
        sl[i].set_wma(wma[i])
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
Function that gets the weighted moving average of a set of prices

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all weighted moving average points for all stocks
"""
def get_wma(df):
    wma = []
    tp = 10
    for i in range(df.shape[1]):
        wma.append(tal.WMA(df[i], timeperiod = tp))
    #return wma

    #extra code to get trend? returns True if bullish, False if bearish
    wma_trends = []
    for i in range(df.shape[1]):
        tempWMA = tal.WMA(df[i], timeperiod = tp)
        wma1 = np.average(tempWMA.to_numpy()[tp-1:int(np.floor(df.shape[0] - tp + 1))])
        wma2 = np.average(tempWMA.to_numpy()[int(np.ceil(df.shape[0] - tp + 1)):])
        wma_trends.append(wma2>wma1)
        print("old: {}, new: {}, result: {}".format(wma1, wma2, wma2>wma1))
    return wma_trends


def main():
    pricesFile = "./prices250.txt"
    prcAll = loadPrices(pricesFile)
    getMyPosition(prcAll)
    #getMyPositionTest_Kenzo(prcAll)

if __name__ == "__main__":
  main()
