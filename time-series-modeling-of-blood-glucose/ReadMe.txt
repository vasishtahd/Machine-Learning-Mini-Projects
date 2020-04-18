 
Commands to run our  Code :

-Taining code:
python datamining_phase.py

-Testing Code:
python testscript.py
Enter Test File Path:<file_path>
Eg :
Enter Test File Path:C:\Users\Admin\PycharmProjects\dataCube\dataminingProject\phase2_data\mealData3.csv
___________________________________________________________________
___________________________________________________________________

The models trained in the training scripts is pickeled. We have attached the pickel files already with this project . 
Therefore you need not retrain the models . We have trained 9 models here. Pickel file name in format


The outputs are stored in pickel files for the respective models.
Here we have enabled to run only our top4 models , though we tried 9 models in our experiment.
The outputs from this is stored in the respective csv file named in the format "output_model<model_no>.csv". 
i.e. model0.pkl, model1.pkl model2.pkl, model3.pkl
___________________________________________________________________
___________________________________________________________________
Indivitual Contribution :
Model 1 :
By Avinash Khatwani
Pickled Model:
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=80, learning_rate='constant',
              learning_rate_init=0.001, max_iter=5000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
In File:model0.pkl
___________________________________________________________________
___________________________________________________________________
Model 2 : By Parantika Ghosh

Pickled Model:
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=100, learning_rate='constant',
              learning_rate_init=0.001, max_iter=5000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
In File:model1.pkl
___________________________________________________________________
___________________________________________________________________
Model 3 : By Shankar Krishnamoorthy

Pickled Model:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=550,
                       n_jobs=-1, oob_score=False, random_state=50, verbose=0,
                       warm_start=False)
In File:model2.pkl
___________________________________________________________________
___________________________________________________________________
Model 4 : By Vasishta Harekal

Pickled Model:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=5, verbose=0,
                       warm_start=False)
In File:model3.pkl
___________________________________________________________________
___________________________________________________________________
Extra Model ##: By Vasishta Harekal, Shankar Krishnamoorthy, Avinash 

Pickled Model:
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)	
In File:model4.pkl
___________________________________________________________________
___________________________________________________________________
Extra Model ##: By Vasishta Harekal, Shankar Krishnamoorthy, Avinash 
Pickled Model:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=18, p=2,
                     weights='uniform')
					 
 				 
In File:model5.pkl
___________________________________________________________________
___________________________________________________________________
Extra Model ##: By Vasishta Harekal, Shankar Krishnamoorthy, Avinash  
Pickled Model:
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=80, learning_rate='constant',
              learning_rate_init=0.001, max_iter=5000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=1, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
In File:model6.pkl
___________________________________________________________________
___________________________________________________________________
Extra Model ##: By Vasishta Harekal, Shankar Krishnamoorthy, Avinash 

Pickled Model:
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
In File:model7.pkl

___________________________________________________________________
___________________________________________________________________
Extra Model ##: By Vasishta Harekal, Shankar Krishnamoorthy, Avinash 
Pickled Model:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=550,
                       n_jobs=-1, oob_score=False, random_state=50, verbose=0,
                       warm_start=False)
In File:model8.pkl
Pickled Model:
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.01,
                   n_estimators=60, random_state=None)
In File:model9.pkl

___________________________________________________________________
___________________________________________________________________
Accuracy Outputs for each model for the given data 
___________________________________________________________________
MLP 1
Min : 0.7083333333333334
Mean : 0.7746563573883163
Max : 0.8350515463917526
_______________________________________________________________________
MLP 2
Min : 0.6666666666666666
Mean : 0.7621993127147767
Max : 0.8041237113402062
_______________________________________________________________________
RandomForest 1
Min : 0.7083333333333334
Mean : 0.7787800687285225
Max : 0.8247422680412371
_______________________________________________________________________
RandomForest 2
Min : 0.7216494845360825
Mean : 0.7829682130584193
Max : 0.8247422680412371
_______________________________________________________________________
SVM
Min : 0.625
Mean : 0.6836734693877551
Max : 0.8367346938775511
_______________________________________________________________________
KNN
Min : 0.5306122448979592
Mean : 0.6511054421768707
Max : 0.7346938775510204
_______________________________________________________________________
MLP
Min : 0.7083333333333334
Mean : 0.7746563573883163
Max : 0.8350515463917526
_______________________________________________________________________
DecisionTree
Min : 0.5625
Mean : 0.6545493197278912
Max : 0.7551020408163265
_______________________________________________________________________
RandomForest 1
Min : 0.7083333333333334
Mean : 0.7787800687285225
Max : 0.8247422680412371
_______________________________________________________________________
Adaboost:
Min : 0.5208333333333334
Mean : 0.6547619047619048
Max : 0.7755102040816326
_______________________________________________________________________