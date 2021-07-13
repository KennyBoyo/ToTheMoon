import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib as tal

"""
Class which holds all stock data for a single stock
"""
class Stock():
    def __init__(self, data):
        self._wma = []
        self._wma2 = []
        self._aroon = []
        self._data = data

    """Setter function for the class which sets its current and previous prices"""
    def set_data(self, data):
        self._data = data

    """Setter function for the class which sets its slow weighted moving average"""
    def set_wma(self, wma):
        self._wma = wma

    """Setter function for the class which sets its fast weighted moving average"""
    def set_wma2(self, wma):
        self._wma2 = wma

    """Setter function for the class which sets its aroon oscillator value"""
    def set_aroon(self, aroon):
        self._aroon = aroon

    """Getter function for the class which returns its current and previous prices"""
    def get_data(self):
        return self._data

    """Getter function for the class which returns the slow weighted moving average"""
    def get_wma(self):
        return self._wma

    """Getter function for the class which returns the fast weighted moving average"""
    def get_wma2(self):
        return self._wma2

    """Getter function for the class which returns the aroon oscillator value for the stock"""
    def get_aroon(self, index):
        return get_aroon(self._data, 25, index)

    """Function which calculates the position based on all indicators"""
    def get_pos(self):
        pos = 0
        wma = self.get_wma()
        wma2 = self.get_wma2()
        grad = get_gradient(wma)[-1]
        grad2 = get_gradient(wma2)[-1]
        sci = get_sign_change_index(wma)
        pos = sci[0]
        aroon = self.get_aroon_pos(sci[1])
        if sci[1] is None:
            return pos * (10000/self.get_data()[0]) * aroon * abs((grad+grad2)/2)
        pos = pos * (10000/self.get_data()[sci[1]]) * aroon * abs((grad+grad2)/2)
        return pos

    """Function which returns the position based on the aroon oscillator value"""
    def get_aroon_pos(self, index):
        th = 0.1
        aroon = get_aroon(self._data, 50, index)[-1]/100
        if abs(aroon) > th:
            pos = 1
        else:
            pos = 0
        return pos

    """Function which returns the maximum position for the stock"""
    def get_def_pos(self):
        return (10000/self.get_data()[-1])

"""
Function which loads prices from a file and returns an array

:param fn: str | string representing the file name to be read from

:return: df | dataframe containing the prices values read from the file
"""
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T

"""
Function that returns a list which represents the slope of the input list

:param list: list | list of datapoints

:return: list | list of length len(list)-1 which represents the gradient of the slope for the datapoints
"""
def get_gradient(list):
    gradList = []
    th = 0.005
    #1 if up, -1 if down
    for i in range(len(list)-1):
        if list[i] < list[i+1] and abs(list[i] - list[i+1])*2/(list[i] + list[i+1]) > th:
            gradList.append(1)
        elif list[i] > list[i+1] and abs(list[i] - list[i+1])*2/(list[i] + list[i+1]) > th:
            gradList.append(-1)
        else:
            gradList.append(0)
    return gradList

"""
Function which takes a list as a parameter and returns the index of the last sign change

:param list: list | list of all the data for the stock

:return: list | first index is trend, and second index is index of sign change
"""
def get_sign_change_index(list):
    trend = 0
    for i in range(len(list)-2):
        if list[-1-i]>list[-2-i]:
            #increasing
            if trend == 0:
                trend = 1
            elif trend == 1:
                continue
            elif trend == -1:
                return [1, len(list) - i]

        elif list[-1-i]<list[-2-i]:
            #decreasing
            if trend == 0:
                trend = -1
            elif trend == 1:
                return [-1, len(list) - i]
            elif trend == -1:
                continue
    return [trend, None]


"""
Final submission function

:param df: list | list of all price values of all stocks to date

:return: np.array | 100 integers denoting daily position
"""
def getMyPosition(prcHistSoFar):
    df = pd.DataFrame(prcHistSoFar).T
    sl = gen_stocks(df)
    positions = []
    for stock in sl:
        positions.append(stock.get_pos())
    return positions


"""
Function that generates a list of stocks from the provided data

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all stocks containing wma and correlation
"""
def gen_stocks(df):
    sl = []
    for i in range(df.shape[1]):
        sl.append(Stock(df[i].values.tolist()))
        sl[i].set_wma(get_wma(df[i], 16))
        sl[i].set_wma2(get_wma(df[i], 8))
    return sl

"""
Function that gets the weighted moving average of a set of prices

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all weighted moving average points for all stocks
"""
def get_wma(df, tp):
    return tal.WMA(df, timeperiod = tp).tolist()

"""
Function that gets the stochastic indicator of the stock. (use over short time period for trigger, 
longer periods for filter)

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all stochastic indicator values
"""
def get_stoch(df, tp):
    stoch = []
    for i in range(df.shape[1]):
        stoch.append(tal.STOCH())

"""
Function that determines the trend using the aroon oscillator and filters the positions which can be taken

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of aroon indicator values
"""
def get_aroon(data, tp, index):
    aroonTemp = []
    array = data[index-tp:index]
    maxindex = array.index(max(array))
    minindex = array.index(min(array))
    aroonhigh = 100 * (float(maxindex)) / tp
    aroonlow = 100 * (float(minindex)) / tp
    aroonTemp.append(float(aroonhigh - aroonlow))
    return aroonTemp

def main():
    pricesFile = "./prices250.txt"
    prcAll = loadPrices(pricesFile)
    getMyPosition(prcAll)

if __name__ == "__main__":
  main()





