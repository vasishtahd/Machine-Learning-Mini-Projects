#################################################
######  Course : 2019Spring-T-EEE591-32406 ######
######  Project : 2                        ######
######  Author : Vasishta Divakar Harekal  ######
#################################################



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


#Loading Dataset
df_submarine = pd.read_csv('sonar_all_data_2.csv',header=None)

print("Data Set Size:",df_submarine.shape)


X,y = df_submarine.iloc[:,:-2].values, df_submarine.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Scaling Data
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) 


#Functiont to generate Confusion Matrix and plot it
def plot_confusion_matrix(ax,y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    else:
        pass


    print("Confusion Matrix:")
    print(cm)


    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.margins(20)
    # Styling  all ticks in the plot ...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    return ax

#Function to check pass the data given through the classifier and test the effectiveness of the classifier
# THis also populates the confusion matrix into the final result figure
def pass_through_classifer( ax,classifier, X_train_std, X_test_std, y_train, y_test ):
    classifier_name=str(classifier).split("(")[0]
    classifier.fit(X_train_std, y_train)
    y_pred = classifier.predict(X_test_std)
    classifier_mis_clasify = (y_test != y_pred).sum()
    classifier_accuracy = accuracy_score(y_test, y_pred)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    y_combined_pred = classifier.predict(X_combined_std)
    cclassifier_accuracy = accuracy_score(y_combined, y_combined_pred)
    print("________________________________________________________________________________\n"
          "\nClassifier:\n{}  \nMisclassified:{}  \nAccuracy:{}  \nCombined Accuracy:{}"
          "\n________________________________________________________________________________\n "
          .format(str(classifier),(classifier_mis_clasify) , str(classifier_accuracy) , str(cclassifier_accuracy))   )

    np.set_printoptions(precision=2)

    class_names =  unique_labels(y_combined, y_combined_pred)

    plot_confusion_matrix(ax , y_test, y_pred, classes=class_names, normalize=True,
                         title=str(classifier_name) )
    return ax


#Generating  PCA vectors from the combining the different existing features
pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#Initialising all the different classifiers
svm = SVC(gamma='scale', decision_function_shape='ovo', )
ppn = Perceptron(max_iter=2000, tol=1e-3, eta0=0.05, fit_intercept=True, random_state=0)
slr = LogisticRegression(solver='liblinear', max_iter =1000,C=10,multi_class='ovr' )
tree = DecisionTreeClassifier(criterion='entropy',max_depth=10 ,random_state=0)
knn = KNeighborsClassifier(n_neighbors=3,p=2,metric='minkowski')
rforest= RandomForestClassifier(n_estimators=100)
classifiers_list=[ppn,slr,tree,rforest,knn,svm]


fig, ax_array = plt.subplots(2,  3, sharex=True, sharey=True)
plt.subplots_adjust(left=.1, bottom=.05, right=.9, top=1, wspace=.2, hspace=.2)

index=0
#Passing the Data through the differnt classifiers and generating confusion matrix for each of them
for ax in np.ravel(ax_array):
    pass_through_classifer(ax, classifiers_list[index], X_train_pca, X_test_pca, y_train, y_test)
    index=index+1

#settting figure size
fig.set_size_inches(9, 6)
plt.show()

