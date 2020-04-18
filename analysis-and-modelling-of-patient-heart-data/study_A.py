#################################################
######  Course : 2019Spring-T-EEE591-32406 ######
######  Project : 1                        ######
######  Author : Vasishta Divakar Harekal  ######
#################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm as cm
 
import seaborn as sns
## function to find_most_highly_correlated_features
def find_most_highly_correlated_features(mydataframe, numtoreport): 
# find the correlations 
    cormatrix = mydataframe.corr() 
# set the correlations on the diagonal or lower triangle to zero, 
# so they will not be reported as the highest ones: 
    cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T 
    cormatrix = cormatrix.stack() 
    cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() 
    cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"] 
    return cormatrix.head(numtoreport)
## Cfunction to find ovariance matrix
def genarate_correlation_matrix(X,cols):
    fig = plt.figure(figsize=(7,7), dpi=100)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet',30)
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)
    major_ticks = np.arange(0,len(cols),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
##    plt.aspect('equal')
    plt.title('Correlation Matrix')
    labels = cols
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=12)
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()
    return(1)
## Function to genearte paiplots
def generate_pair_plotting_figures(df):
    sns.set(style='whitegrid', context='notebook')
    cols = df.columns
    sns.pairplot(df[cols],size=2.5)
    plt.show()
 
heart_data = pd.read_csv('heart1.csv')
 
cols = heart_data.columns
X = heart_data.iloc[:,0:12].values
Y = heart_data.iloc[:,13].values
 
# 
print(' Descriptive Statistics ')
print(heart_data.describe())
## Genarating  Highly Correlated Lists
print("10 Most Highly Correlated Variables ")
top_correlations=find_most_highly_correlated_features(heart_data,10)
closest_y_correlated=[]
index=0
 
print( top_correlations)
  
closest_y_correlated= top_correlations.loc[top_correlations['SecondVariable'] == cols[-1]]

important_correlations=closest_y_correlated["FirstVariable"].tolist()[:6]

print("Most Correlated variables for Y:"+ str( closest_y_correlated["FirstVariable"].tolist()[:6]))        
## heat plot of covariance
print(' Covariance Matrix for all variables')
genarate_correlation_matrix(heart_data.iloc[:,0:14],cols[0:14])
## Pair plotting
print(' Pair plotting  for all variables ')
generate_pair_plotting_figures(heart_data)
 