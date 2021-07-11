import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot
import csv
import talib as tal

y = []
x = []

"""Finds the Schaff trend cycle and returns an indicator"""
def STC(prices):
    macd, signal, hist = tal.MACD(prices,
                                  fastperiod=23,
                                  slowperiod=50,
                                  signalperiod=10)

    k, d = tal.STOCH(high=macd,
                     low=macd,
                     close=macd,
                     fastk_period=10,
                     slowk_period=10)

    macdValue = macd[-1] * 100
    kValue = k[-1]
    dValue = d[-1]

    if 0 <= dValue - kValue < 0.01:
        return 1
    if 0 <= kValue - dValue < 0.01:
        return -1

    schaff = (macdValue - kValue) / (dValue - kValue) / 100

    if schaff > 100:
        return 1
    elif schaff < -100:
        return -1

    return schaff


"""Function called by eval.py. Param: float[stock][day]"""
def getMyPosition(prcHistSoFar):
    global y, x
    newPos = [0 for _ in range(100)]
    indicator = []
    for stock in range(100):
        stockCurrentPrice = prcHistSoFar[stock][-1]
        indicator.append(STC(prcHistSoFar[stock]))
        if 20 < abs(indicator[-1]) < 80:
            newPos[stock] = stockCurrentPrice * (1 + indicator[-1] / 100)
    y.append(np.average(indicator))
    x.append(np.average(newPos))
    return newPos


def plot():
    global y, x
    pyplot.plot(y)
    pyplot.plot(x)
    pyplot.show()
