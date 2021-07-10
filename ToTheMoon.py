#hehe xd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import talib as tal

# TESTING

"""
Class which holds all stock data for a single stock
"""
class Stock():
    def __init__(self, data):
        self._wma = []
        self._correl = []
        self._aroon = []
        self._data = data

    def set_data(self, data):
        self._data = data

    def set_correl(self, correl):
        self._correl = correl

    def set_wma(self, wma):
        self._wma = wma

    def set_aroon(self, aroon):
        self._aroon = aroon

    def get_data(self):
        return self._data

    def get_correl(self):
        return self._correl

    def get_max_correl_index(self):
        return self._correl.index(sorted(self._correl)[-2])

    def get_wma(self):
        return self._wma

    def get_aroon(self):
        return self._aroon

    def get_wma_pos(self):
        #add your calcs here
        pos = 0
        wma = self.get_wma()
        sci = get_sign_change_index(wma)
        print(sci)
        pos = sci[0]
        """if self.get_wma():
            pos = -1
        else:
            pos = 1"""
        pos = pos * (10000/self.get_data()[sci[1]])
        return pos

    def get_aroon_pos(self):
        #add your calcs here
        th = 70
        aroon = self.get_aroon()
        if aroon[-1] > th:
            pos = 1
        elif aroon[-1] < -th:
            pos = -1
        else:
            pos = 0
        pos = pos * (10000/self.get_data()[-1])
        return pos

    def get_def_pos(self):
        return (10000/self.get_data()[-1])


def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T


def get_sign_change_index(list):
    trend = 0
    #print(list)
    for i in range(len(list)):
        #print(list[-1-i])
        #print(list[-2-i])
        #print(list[-1-i]>list[-2-i])
        #print(trend)
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


"""
Final submission function

:param df: list | list of all price values of all stocks to date

:return: np.array | 100 integers denoting daily position
"""
def getMyPosition(prcHistSoFar):
    #df = get_data('prices250.txt')
    df = pd.DataFrame(prcHistSoFar).T
    #print(df[0].describe())
    sl = gen_stocks(df)
    #for s in sl:
        #print(s.get_max_correl_index())
        #print(s.get_wma())
    positions = []
    for stock in sl:
        positions.append(stock.get_wma_pos())
        #positions.append(stock.get_aroon_pos())
        #positions.append(stock.get_def_pos())
    #for i in range(len(sl)):
        #if i == 2:
        #    positions.append(sl[i].get_pos())
        #else:
        #    positions.append(0)
    #print(positions)
    return positions

"""
Function used to test implementation of new functons
"""
def getMyPositionTest_Kenzo(prcHistSoFar):
    #df = get_data('prices250.txt')
    df = pd.DataFrame(prcHistSoFar).T
    #print(df[0])
    c = get_correl(df)
    positions = []
    #print(len(c))
    #for i in c:
    #    print("index1={}, index2={}, correlation={}, length"
    #          .format(i.index(1), i.index(sorted(i)[-2]), sorted(i)[-2]), len(i))
    #print(df[0].describe())
    sl = gen_stocks(df)
    for stock in sl:
        #print(stock.get_wma())
        print(stock.get_max_correl_index())
        #print(stock.get_aroon())
        plt.subplot(2,1,1)
        plt.plot(stock.get_data())
        plt.plot(stock.get_wma())
        plt.subplot(2,1,2)
        plt.plot(stock.get_aroon())
        plt.show()
        positions.append(stock.get_wma_pos())
    return positions

"""
Function that generates a list of stocks from the provided data

:param df: pd.Dataframe | Dataframe values containing all prices to date of all stocks

:return: list | list of all stocks containing wma and correlation
"""
def gen_stocks(df):
    sl = []
    c = get_correl(df)
    #wma = get_wma(df)
    #aroon = get_aroon(df)
    for i in range(df.shape[1]):
        sl.append(Stock(df[i].to_numpy()))
        sl[i].set_correl(c[i])
        #sl[i].set_wma(wma[i])
        sl[i].set_wma(get_wma(df[i], 10))
        sl[i].set_aroon(get_aroon(df[i], 25))
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
def get_wma(df, tp):
    return tal.WMA(df, timeperiod = tp).tolist()


    wma = []
    tp = 10
    tp2 = 20
    for i in range(df.shape[1]):
        wma.append(tal.WMA(df[i], timeperiod = tp))
    return wma

    #extra code to get trend? returns True if bullish, False if bearish
    wma_trends = []
    for i in range(df.shape[1]):
        tempWMA = tal.WMA(df[i], timeperiod = tp)
        #tempWMA2 = tal.WMA(df[i], timeperiod=tp2)
        #plt.plot(tempWMA)
        #plt.plot(tempWMA2)
        #plt.plot(df[i].values)
        #plt.title(i)
        #plt.show()

        wma1 = np.average(tempWMA.to_numpy()[tp-1:int(np.floor(df.shape[0] - tp + 1))])
        wma2 = np.average(tempWMA.to_numpy()[int(np.ceil(df.shape[0] - tp + 1)):])
        wma_trends.append(wma2>wma1)
        #print("old: {}, new: {}, result: {}".format(wma1, wma2, wma2>wma1))
    return wma_trends

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
def get_aroon(df, tp):
    aroonTemp = []
    for j in range(df.shape[0]):
        if j >= tp:
            #print(df.index(max(df[i][(j-14):j])))
            #print(df.index(min(df[i][(j-14):j])))
            array = df[(j-tp):j].values.tolist()
            maxindex = array.index(max(array))
            minindex = array.index(min(array))
            aroonhigh = 100*(float(maxindex))/tp
            aroonlow = 100*(float(minindex))/tp
            aroonTemp.append(float(aroonhigh-aroonlow))
            #print(array)


    """fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(-100, 101, 25)
    minor_ticks = np.arange(-100, 101, 10)

    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


    plt.plot(aroonTemp)
    ax2 = fig.add_subplot(2, 1, 2)
    plt.plot(df[i].values.tolist())
    plt.title(i)
    plt.show()"""
    #print(aroonTemp)
    return aroonTemp



# Conventional main python script setup, also testing
def main():
    pricesFile = "./prices250.txt"
    prcAll = loadPrices(pricesFile)
    #getMyPosition(prcAll)
    getMyPositionTest_Kenzo(prcAll)

if __name__ == "__main__":
  main()





