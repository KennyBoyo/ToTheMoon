import numpy as np
import pandas as pd
import matplotlib
import csv
import talib as tal


# Parameter format: float[stock][day]
def getMyPosition(prcHistSoFar):
    return [9000 for _ in range(100)]


# For testing first occurrence (day 200)
def main():
    prcAll = eval.loadPrices(eval.pricesFile)
    getMyPosition(prcAll)


if __name__ == "__main__":
    main()