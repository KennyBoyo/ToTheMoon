import numpy
import numpy as np
import pandas as pd
import matplotlib
import csv
import talib as tal


def findDiffAvg(stockHis):
    totalDiff = 0
    diffLen = len(stockHis) - 1
    for day in range(diffLen):
        totalDiff += (stockHis[day + 1] - stockHis[day])
    return totalDiff / diffLen


# Parameter format: float[stock][day]
def getMyPosition(prcHistSoFar):
    newPos = [0 for _ in range(100)]
    for stock in range(100):
        stockCurrent = prcHistSoFar[stock][-1]
        stockAvg = numpy.average(prcHistSoFar[stock])
        stockDiff = findDiffAvg(prcHistSoFar[stock])
        if (stockDiff > 0 and stockCurrent - stockAvg > 0 or
            stockDiff < 0 and stockCurrent - stockAvg < 0):
            newPos[stock] = stockCurrent ** 2 / stockAvg
        else:
            newPos[stock] = stockCurrent
    return newPos


# For testing first occurrence (day 200)
def main():
    prcAll = eval.loadPrices(eval.pricesFile)
    getMyPosition(prcAll)


if __name__ == "__main__":
    main()