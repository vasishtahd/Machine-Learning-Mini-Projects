#################################################
######  Course : 2019Spring-T-EEE591-32406 ######
######  Project : 1b                       ######
######  Author : Vasishta Divakar Harekal  ######
#################################################

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
 
#['thal', 'nmvcf', 'eia', 'mhr', 'opst', 'cpt']
################ Loading Data #######################################################################  
heart_data = pd.read_csv('heart1.csv')
################ Including all parameters for Machine Learning####################################### 
X = heart_data.loc[ :,['thal', 'nmvcf', 'eia', 'mhr', 'opst', 'cpt', 'dests','age','sex','rbp','sc','rer','fbs', ]].values.tolist()
 

y = np.ravel(heart_data.loc[ :,['a1p2']].values.tolist())
 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

################ Scaling #############################################################  
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

################ Perceptron########################################################## 
ppn = Perceptron(max_iter=200, tol=1e-3, eta0=0.0001, fit_intercept=True, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
ppn_mis_clasify=(y_test != y_pred).sum()
ppn_accuracy=accuracy_score(y_test, y_pred)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = ppn.predict(X_combined_std)
cppn_accuracy=accuracy_score(y_combined, y_combined_pred)
 
################ Logistic Regression#################################################
from sklearn.linear_model import LogisticRegression
slr = LogisticRegression(solver='liblinear', max_iter =1000,C=10,multi_class='ovr' )
slr.fit(X_train_std,y_train)
y_pred = slr.predict(X_test_std)
slr_mis_clasify =(y_test != y_pred).sum() 
slr_accuracy=accuracy_score(y_test, y_pred)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = slr.predict(X_combined_std)
cslr_accuracy=accuracy_score(y_combined, y_combined_pred)

################ Support Vector Machine ##########################################
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1, random_state=10)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
svm_mis_clasify= (y_test != y_pred).sum() 
svm_accuracy=accuracy_score(y_test, y_pred)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = svm.predict(X_combined_std)
csvm_accuracy=accuracy_score(y_combined, y_combined_pred)


################ Decision Tree Classifier ##########################################
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3 ,random_state=0)
tree.fit(X_train_std,y_train)
y_pred = tree.predict(X_test_std)
dtc_mis_clasify= (y_test != y_pred).sum()
dtc_accuracy= accuracy_score(y_test, y_pred)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = tree.predict(X_combined_std)
cdtc_accuracy=accuracy_score(y_combined, y_combined_pred)

################ KNeighbors Classifier #############################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)
y_pred = knn.predict(X_test_std)
knn_mis_clasify=(y_test != y_pred).sum()
knn_accuracy= accuracy_score(y_test, y_pred)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
y_combined_pred = knn.predict(X_combined_std)
cknn_accuracy=accuracy_score(y_combined, y_combined_pred)

################################ Displaying Results ################################ 

print("  |\t|Classifier      |\t|     Misclassified      |\t| Accuracy            |\t|  Combined Accuracy    |\t|"+
      "\n|\t| Perceptron     |\t|"+str(ppn_mis_clasify)+"|\t|"+str(ppn_accuracy)+"|\t|"+str(cppn_accuracy)+"|\t|"+
      "\n|\t|Linear Regress. |\t|"+str(slr_mis_clasify)+"|\t|"+str(slr_accuracy)+"|\t|"+str(cslr_accuracy)+"|\t|"+
      "\n|\t| SVM            |\t|"+str(svm_mis_clasify)+"|\t|"+str(svm_accuracy)+"|\t|"+str(csvm_accuracy)+"|\t|"+
      "\n|\t|Decision Tree   |\t|"+str(dtc_mis_clasify)+"|\t|"+str(dtc_accuracy)+"|\t|"+str(cdtc_accuracy)+"|\t|"+
      "\n|\t| KNeighbhors    |\t|"+str(knn_mis_clasify)+"|\t|"+str(knn_accuracy)+"|\t|"+str(cknn_accuracy)+"|\t|"
      )
