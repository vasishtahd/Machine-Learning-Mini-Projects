import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.fftpack
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pywt

df = pd.read_csv('datamining_data/CGMDatenumLunchPat4.CSV')
df11 = pd.read_csv('datamining_data/CGMSeriesLunchPat4.CSV', skiprows=[29])
# df2 = pd.read_csv('InsulinDatenumLunchPat1.CSV')
# df3 = pd.read_csv('InsulinBasalLunchPat1.CSV')
# df4 = pd.read_csv('InsulinBolusLunchPat1.CSV')

df = df.interpolate(axis=1, method='linear')
df1 = df11.interpolate(axis=1, method='linear')

featureMatrixVector = []
rmsVector = []
for i in range(len(df) - 1):
    # for i in range(0,1):
    featureVector = []
    cgmDatenumArr = np.array(df.iloc[i, :].tolist())
    cgmDatenumArr = cgmDatenumArr[::-1]
    cgmValuesArr = np.array(df1.iloc[i, :].tolist())
    cgmValuesArr = cgmValuesArr[::-1]

    plt.plot(cgmDatenumArr, cgmValuesArr)
    plt.title("cgm vs time")
    # plt.show()

    cgmValuesArr31 = pd.Series(cgmValuesArr[1:30])
    pct = cgmValuesArr31.pct_change()
    pct = pct.tolist()
    plt.plot(cgmDatenumArr[1:30], pct)
    plt.title("pct(rate of chage) vs time")
    plt.axhline(y=0, color='r', linestyle='-')
    # plt.show()
    cgmValuesArr312 = pd.Series(cgmValuesArr[1:30])
    iqr = cgmValuesArr312.quantile([.25, .75])
    print("iqr[0.25]\n", iqr[0.25])
    print("iqr[0.75]\n", iqr[0.75])



    rolling_mean = cgmValuesArr312.rolling(window=3,min_periods=1).mean()
    rolling_std = cgmValuesArr312.rolling(window=3,min_periods=1).std()

    print("rolling_mean", rolling_mean.values)
    print("rolling_std", rolling_std.values)
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    # plt.show()


    pctMaxInd = pct.index(np.nanmax(pct));
    ###  Taking max PCT index
    pctZeroInd = 0;
    for i in range(pctMaxInd, -1, -1):
        if (pct[i] <= 0):
            pctZeroInd = i;

    cgmFFTValues = abs(scipy.fftpack.fft(cgmValuesArr[1:31]))
    plt.stem(cgmDatenumArr[2:28], cgmFFTValues[2:28], use_line_collection="true")
    # plt.show()

    #    Calculate RMS
    rms = np.sqrt(np.mean(cgmValuesArr ** 2))
    rmsVector.append(rms)
    rmsVectorPd = pd.DataFrame(rmsVector)
    rmsVectorPd.fillna(0)

    print("RMS", rms)

    #    Calculate DWT
    (ca, cd) = pywt.dwt(cgmValuesArr, 'sym4')

    featureVector.append(cgmValuesArr31[pctMaxInd])
    featureVector.append(cgmValuesArr31[pctZeroInd])

    featureVector.append(iqr[0.25])
    featureVector.append(iqr[0.75])
    featureVector.extend(rolling_mean.values[1:])
    featureVector.extend(rolling_std.values[1:])



    for i in range(2, 9):
        featureVector.append(cgmFFTValues[i])
    featureVector.append(rms)
    featureMatrixVector.append(featureVector)

    featureMatrixVectorPd = pd.DataFrame(featureMatrixVector)
    featureMatrixVectorPd.fillna(0)

    numpyFeatureMatrix = np.array(featureMatrixVectorPd)


print(numpyFeatureMatrix.shape)

# plot RMS for 1 Patient(33 rows)
plt.scatter(list(range(1, 30)), rmsVector[1:30])
plt.title("RMS for Patient")
plt.show()

pca = PCA(n_components=5)
pca.fit(numpyFeatureMatrix)
pcaComp = pca.components_
pcaExpVariance = pca.explained_variance_
print("PCA Components= ", pcaComp)
print("PCA variance= ", pcaExpVariance)
pcaTransformed = pca.transform(numpyFeatureMatrix)
# numpyFeatureMatrix1 = numpyFeatureMatrix*pcaComp
# numpyFeatureMatrix1 = numpyFeatureMatrix*pcaTransformed
# print(B)

# bar Plot for eigen vector 1 row
# plt.bar(list(range(1,10)),pcaComp[1,:])
# plt.title("eigen vector-")
# plt.show()

# PLOT EigenVectors
barWidth = 0.25
r1 = list(range(0, 68))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]



plt.bar(r1, pcaComp[0, :], width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, pcaComp[1, :], width=barWidth, edgecolor='white', label='var1')
plt.bar(r3, pcaComp[2, :], width=barWidth, edgecolor='white', label='var1')
plt.bar(r4, pcaComp[3, :], width=barWidth, edgecolor='white', label='var1')
plt.bar(r5, pcaComp[4, :], width=barWidth, edgecolor='white', label='var1')
plt.show()

# Plot Eigen Values
plt.bar(list(range(0, 5)), pcaExpVariance)
plt.show()

# PCA results for Patient 1
plt.scatter(list(range(0, 50)), pcaTransformed[0:50, 0])
plt.title("PCA-1")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 50)), pcaTransformed[0:50, 1])
plt.title("PCA-2")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 50)), pcaTransformed[0:50, 2])
plt.title("PCA-3")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 50)), pcaTransformed[0:50, 3])
plt.title("PCA-4")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()
plt.scatter(list(range(0, 50)), pcaTransformed[0:50, 4])
plt.title("PCA-5")
plt.ylabel("Feature Vectors")
plt.xlabel("Time")
plt.show()