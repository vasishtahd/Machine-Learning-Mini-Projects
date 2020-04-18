 
 
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
from sklearn.svm import SVC

def loadData():
#READ MealMeal data
    dfDataMeal = pd.DataFrame();
    for i in range(5):
        rows = []
        with open('phase2_data/mealData'+str(i+1)+'.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                if(len(row)<30):
                    pass
                else:
                    rows.append(row)
            frame = pd.DataFrame(rows)
            dfDataMeal = dfDataMeal.append(frame, ignore_index='true')
    dfDataMeal = dfDataMeal.iloc[:, 0:30]
    dfDataMeal = dfDataMeal.astype(float)
    dfDataMeal = cleanData(dfDataMeal)
    dfDataMeal = addLabel(dfDataMeal, 'meal')
#    dfDataMeal['yLabel'] = dfDataMeal['yLabel'].astype(int)
    

#READ noMeal data
    dfDataNoMeal = pd.DataFrame();
    for i in range(5):
        dfDataTemp = pd.read_csv("phase2_data/Nomeal"+str(i+1)+".csv", usecols = [i for i in range(30)], header = None)
        dfDataNoMeal = dfDataNoMeal.append(dfDataTemp, ignore_index='true')
    dfDataNoMeal = cleanData(dfDataNoMeal)
    dfDataNoMeal = addLabel(dfDataNoMeal, 'noMeal')
    
    data = combineData(dfDataMeal, dfDataNoMeal)

    print("NoMeal Data:\n"+str(dfDataNoMeal) +"Meal Data:\n" +str(dfDataMeal))


    return data

def loadTestData():
    fileinput = str(input("input file with path "))

    # READ noMeal data
    dfDataNoMeal = pd.DataFrame();
    dfDataTemp = pd.read_csv(fileinput, usecols=[i for i in range(30)],
                                 header=None)
    dfDataNoMeal = dfDataNoMeal.append(dfDataTemp, ignore_index='true')
    dfDataNoMeal = cleanData(dfDataNoMeal)
    # dfDataNoMeal = addLabel(dfDataNoMeal, 'noMeal')

    return dfDataNoMeal

def cleanData(data):
    data = data.dropna(how = 'all')
    data = data.interpolate(method = 'linear', limit_direction = 'both')
    return data

def combineData(dfDataMeal, dfDataNoMeal):
    data = dfDataMeal.append(dfDataNoMeal, ignore_index='true')
    return data

def addLabel(dataFrame, meal):
    if(meal == 'meal'):
        mealCol = np.empty(dataFrame.shape[0])
        mealCol.fill(1)
        dataFrame['yLabel'] = mealCol
        return dataFrame
    elif(meal == 'noMeal'):
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
    (ca, cd) = pywt.dwt(dfData,'sym4')
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
        rmsRow = np.sqrt(np.mean(row**2))
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

    rolling_mean = scaledDataDF.rolling(window=5,min_periods=1).mean()
    rolling_std = scaledDataDF.rolling(window=5,min_periods=1).std()

    print(rolling_std)

    print(rolling_mean)
    result = pd.concat([pctDf, caDwtDf, varianceDF,rolling_mean,rolling_std ], axis=1, sort=False)
    result=result.dropna()
    return(result)

def kFoldSvm(X, y):
    scores = []
    accuracy = []
    best_svr = svm.SVC(gamma='scale')
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        best_svr.fit(X_train, y_train)
        predicted = best_svr.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        scores.append(best_svr.score(X_test, y_test))

    print("SVM")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))
    return best_svr

def kFoldKnn(X,y):
    scores = []
    accuracy = []
    knn = KNeighborsClassifier(n_neighbors=18, algorithm='auto', leaf_size=30)
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        knn.fit(X_train, y_train)
        predicted = knn.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        scores.append(knn.score(X_test, y_test))

    print("KNN")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))

    return knn

def kFoldMlp(X,y):
    scores = []
    accuracy = []
    mlp = MLPClassifier(hidden_layer_sizes=(80), max_iter=5000,activation = 'relu',solver='adam',random_state=1)
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        mlp.fit(X_train, y_train)
        predicted = mlp.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        scores.append(mlp.score(X_test, y_test))

    print("MLP")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))

    return mlp

def kFoldMlp2(X,y):
    scores = []
    accuracy = []
    mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=5000,activation = 'relu',solver='adam',random_state=1)
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        mlp.fit(X_train, y_train)
        predicted = mlp.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        scores.append(mlp.score(X_test, y_test))

    print("MLP")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))

    return mlp

def kFoldRandomForest(X,y):
    scores = []
    accuracy = []
    clf=RandomForestClassifier(n_estimators=550, max_features='auto', n_jobs = -1,random_state =50)
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
            test_index]
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        scores.append(clf.score(X_test, y_test))

    print("RandomForest 1")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))

    return clf
def kFoldRandomForest2(X,y):
    scores = []
    accuracy = []
    clf=RandomForestClassifier(n_estimators=200, max_features='auto', n_jobs = -1,random_state =5)
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
            test_index]
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        scores.append(clf.score(X_test, y_test))

    print("RandomForest 2")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))

    return clf


def kFoldDecisionTree(X,y):
    scores = []
    accuracy = []
    model = tree.DecisionTreeClassifier()
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
            test_index]
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        scores.append(model.score(X_test, y_test))

    print("DecisionTree")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))

    return model

def kFoldAdaboost(X,y):
    scores = []
    accuracy = []
    abc = AdaBoostClassifier(n_estimators=60, learning_rate=0.01)

    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
            test_index]
        model = abc.fit(X_train, y_train)
        predicted = model.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, predicted))

        scores.append(model.score(X_test, y_test))

    print("Adaboost:")
    print("Min : " + str(np.min(accuracy)))
    print("Mean : "+ str(np.mean(accuracy)))
    print("Max : "+ str( np.max(accuracy)))

    return abc

def main():
    dfData = loadData()
    y = dfData.iloc[:, -1]
    dfData = dfData.iloc[:,0:30]
    scaledData = stdScale(dfData)
    scaledDataDF = pd.DataFrame(data=scaledData)
    print("_______________________________________________________________________")
    X = calcFeaturesDF(scaledDataDF, scaledData)
    print("_______________________________________________________________________")
    clfs=[]
    clfs.append(kFoldMlp(X, y))
    print("_______________________________________________________________________")
    clfs.append(kFoldMlp2(X, y))
    print("_______________________________________________________________________")
    clfs.append(kFoldRandomForest(X, y))
    print("_______________________________________________________________________")
    clfs.append(kFoldRandomForest2(X, y))
    print("_______________________________________________________________________")
    clfs.append(kFoldSvm(X, y))
    print("_______________________________________________________________________")
    clfs.append(kFoldKnn(X,y))
    print("_______________________________________________________________________")
    clfs.append(kFoldMlp(X,y))
    print("_______________________________________________________________________")
    clfs.append(kFoldDecisionTree(X,y))
    print("_______________________________________________________________________")
    clfs.append(kFoldRandomForest(X,y))
    print("_______________________________________________________________________")
    clfs.append(kFoldAdaboost(X,y))
    print("_______________________________________________________________________")

    for index  in range(len(clfs)):

        with open("model"+str(index)+".pkl", 'wb') as file:
            pickle.dump(clfs[index], file)
        # s = pickle.dumps(clfs[index],  open("model"+str(index)+".pkl", "wb"))

        print("Pickled Model:\n"+str(clfs[index])+"\nIn File:"+"model"+str(index)+".pkl")
    #
    # dfTestData = loadTestData()
    # scaledTestData = stdScale(dfTestData)
    # scaledTestDataDF = pd.DataFrame(data=scaledTestData)
    #
    # XTest = calcFeaturesDF(scaledTestDataDF, scaledTestData)
    # print(XTest)
    # for index  in range(len(clfs)):
    #     predicted = clfs[index].predict(XTest)
    #     print(predicted)


#    scaledData = pd.DataFrame(scaledData)
#    scaledData['rms'] = rms

if __name__ == '__main__':
    main()