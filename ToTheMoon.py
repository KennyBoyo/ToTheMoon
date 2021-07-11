import numpy
import numpy as np
import pandas as pd
import matplotlib
import csv
import talib as tal


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

    macdValue = macd[-1]
    kValue = k[-1]
    dValue = d[-1]

    if kValue == dValue:
        return 0

    schaff = ((macdValue - kValue) / (dValue - kValue)) / 100
    return schaff


"""Function called by eval.py. Param: float[stock][day]"""
def getMyPosition(prcHistSoFar):
    newPos = [0 for _ in range(100)]
    for stock in range(100):
        stockCurrentPrice = prcHistSoFar[stock][-1]
        schaff = STC(prcHistSoFar[stock])
        newPos[stock] = stockCurrentPrice * (1 + schaff)
    return newPos
