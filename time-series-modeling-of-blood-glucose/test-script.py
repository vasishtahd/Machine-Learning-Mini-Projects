 
#
# Commands to run our  Code :
#
# -Taining code:
# python datamining_phase.py
#
# -Testing Code:
# python testscript.py
# Enter Test File Path:<file_path>
 

import pandas as pd
import numpy as np
import csv

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics
from numpy import trapz
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import scipy.fftpack
import statistics
import pickle



def loadTestData():
    fileinput = str(input("Enter Test File Path:"))

    # READ noMeal data
    dfDataNoMeal = pd.DataFrame();
    dfDataTemp = pd.read_csv(fileinput, usecols=[i for i in range(30)],
                                 header=None)
    dfDataNoMeal = dfDataNoMeal.append(dfDataTemp, ignore_index='true')
    dfDataNoMeal = cleanData(dfDataNoMeal)
    # dfDataNoMeal = addLabel(dfDataNoMeal, 'noMeal')

    return dfDataNoMeal


def cleanData(data):
    data = data.dropna(how='all')
    data = data.interpolate(method='linear', limit_direction='both')
    return data


def combineData(dfDataMeal, dfDataNoMeal):
    data = dfDataMeal.append(dfDataNoMeal, ignore_index='true')
    return data


def addLabel(dataFrame, meal):
    if (meal == 'meal'):
        mealCol = np.empty(dataFrame.shape[0])
        mealCol.fill(1)
        dataFrame['yLabel'] = mealCol
        return dataFrame
    elif (meal == 'noMeal'):
        noMealCol = np.empty(dataFrame.shape[0])
        noMealCol.fill(0)
        dataFrame['yLabel'] = noMealCol
        return dataFrame


def stdScale(dfData):
    stdData = StandardScaler().fit_transform(dfData)
    return stdData


def calcPct(rowData):
    series = pd.Series(rowData)
    pctData = series.pct_change()
    return pctData


def calcAucDF(dfData):
    aucArr = trapz(dfData, dx=1)
    dfAuc = pd.DataFrame(aucArr)
    return dfAuc


def calcDwtDF(dfData):
    (ca, cd) = pywt.dwt(dfData, 'sym4')
    caDF = pd.DataFrame(ca)
    cdDF = pd.DataFrame(cd)
    caCdDF = pd.concat([caDF, cdDF], axis=1, sort=False)
    return caCdDF


def calcPctDF(scaledData):
    pct = []
    for row in scaledData:
        pctRow = calcPct(row)
        pctRow = pctRow.to_numpy()
        min1 = sorted(pctRow)[0]
        min2 = sorted(pctRow)[1]
        max1 = sorted(pctRow)[-1]
        max2 = sorted(pctRow)[-2]

        minMaxPctRow = []
        minMaxPctRow.append(min1)
        minMaxPctRow.append(min2)
        minMaxPctRow.append(max1)
        minMaxPctRow.append(max2)

        pct.append(minMaxPctRow)
    dfObj = pd.DataFrame(pct)
    dfObj = dfObj.iloc[:, 1:30]
    return dfObj


def calcRmsDF(scaledData):
    rms = []
    for row in scaledData:
        rmsRow = np.sqrt(np.mean(row ** 2))
        rms.append(rmsRow)
    rms = pd.DataFrame(rms)
    return rms


def calcVarDF(scaledData):
    var = []
    for row in scaledData:
        varRow = statistics.variance(row)
        var.append(varRow)
    var = pd.DataFrame(var)
    return var


def calcFftDF(scaledData):
    fft = []
    for row in scaledData:
        fftRow = abs(scipy.fftpack.fft(row))
        fftRowMin = fftRow.min()
        fftRowMax = fftRow.max()
        fftRowMean = fftRow.mean()
        fftRow1 = []
        fftRow1.append(fftRowMax)
        fftRow1.append(fftRowMin)
        fftRow1.append(fftRowMean)
        fft.append(fftRow1)
    fft = pd.DataFrame(fft)
    return fft


def calcMinMax(dfData):
    dfData = dfData.to_numpy()
    minMax = []
    for row in dfData:
        min1 = sorted(row)[0]
        min2 = sorted(row)[1]
        max1 = sorted(row)[-1]
        max2 = sorted(row)[-2]
        minMaxRow1 = []
        minMaxRow1.append(min1)
        minMaxRow1.append(min2)
        minMaxRow1.append(max1)
        minMaxRow1.append(max2)
        minMax.append(minMaxRow1)
    minMax = pd.DataFrame(minMax)
    return minMax


def calcFeaturesDF(scaledDataDF, scaledData):
    dfAuc = calcAucDF(scaledDataDF)
    caDwtDf = calcDwtDF(scaledDataDF)
    minMax = calcMinMax(scaledDataDF)
    pctDf = calcPctDF(scaledData)
    rmsDF = calcRmsDF(scaledData)
    fftDF = calcFftDF(scaledData)
    varianceDF = calcVarDF(scaledData)

    rolling_mean = scaledDataDF.rolling(window=5, min_periods=1).mean()
    rolling_std = scaledDataDF.rolling(window=5, min_periods=1).std()

    print(rolling_std)

    print(rolling_mean)
    result = pd.concat([pctDf, caDwtDf, varianceDF, rolling_mean, rolling_std], axis=1, sort=False)
    result = result.dropna()
    return (result)


def prediction_method():
    dfTestData = loadTestData()
    scaledTestData = stdScale(dfTestData)
    scaledTestDataDF = pd.DataFrame(data=scaledTestData)

    XTest = calcFeaturesDF(scaledTestDataDF, scaledTestData)
    
    for index in range(4):
        clf=""
        with open("model"+str(index)+".pkl", 'rb') as file:
            clf = pickle.load(file)



        print("\n________________________________________________________________\n"+
              "Fetched  Model:\n" + str(clf) + "\n\nFrom File: " + "model" + str(index) + ".pkl")

        predicted = clf.predict(XTest)
        output_array = np.asarray(predicted)
        np.savetxt("output_model"+str(index)+".csv", output_array, delimiter=",")

        print("Predicted Labels:\n\n " + str(
            predicted) +"\n\nModel Saved to CSV: "+ "output_model"+str(index)+".csv" +
              "\n________________________________________________________________\n")



if __name__ == '__main__':
    prediction_method()