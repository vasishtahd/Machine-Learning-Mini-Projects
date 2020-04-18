 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

"""
Scaler Object to scale feature space.
"""
scaler = MinMaxScaler()

"""
Separating Features and Output for 2015 dataset.
Normalisiation of the feature space is done.
"""
df_2015 = pd.read_csv("2015.csv")
y_2015 = df_2015.iloc[:,3].values
features_2015 = scaler.fit_transform(df_2015.iloc[:,5:])

"""
We use a MLP with 1 hidden layer, 40 hidden nodes.
"""
 
print("~~~~~~~~2015~~~~~~~~")

### Splitting Combined data

X_2015_train, X_2015_test, y_2015_train, y_2015_test = train_test_split(features_2015, y_2015, test_size=0.33, random_state=42)


# Set the parameters by cross-validation
tuned_parameters = [{'hidden_layer_sizes':[(100, 2),(1, 2), (10, 2),(30,2),(50,2) ],
                     'max_iter':[1000],
                     'activation' : ['identity', 'tanh', 'relu'],
                     'alpha' : [0.0001,0.02],
                     'solver' : [   'sgd', 'adam'],
                     }]



clf = GridSearchCV(MLPRegressor(), tuned_parameters)
clf.fit(X_2015_train, y_2015_train)

print(clf.best_params_)
y_true, y_pred = y_2015_test, clf.predict(X_2015_test)
print("Mean squared error MLP for 2015: %.2f "% sqrt(mean_squared_error(y_true, y_pred)))
 

"""
Separating Features and Output for 2016 dataset.
Normalisiation of the feature space is done.
"""
print("~~~~~~~~2016~~~~~~~~")
df_2016 = pd.read_csv("2016.csv")
y_2016 = df_2016.iloc[:,3].values
features_2016 = scaler.fit_transform(df_2016.iloc[:,6:])

### Splitting Combined data

X_2016_train, X_2016_test, y_2016_train, y_2016_test = train_test_split(features_2016, y_2016, test_size=0.33, random_state=42)


# Set the parameters by cross-validation
tuned_parameters = [{'hidden_layer_sizes':[(100, 2),(1, 2), (10, 2),(30,2),(50,2) ],
                     'max_iter':[1000],
                     'activation' : ['identity', 'tanh', 'relu'],
                     'alpha' : [0.0001,0.02],
                     'solver' : [   'sgd', 'adam'],
                     }]
 
 
clf = GridSearchCV(MLPRegressor(), tuned_parameters)
clf.fit(X_2016_train, y_2016_train)
y_true, y_pred = y_2016_test, clf.predict(X_2016_test)
print("Mean squared error MLP for 2016: %.2f "% sqrt(mean_squared_error(y_true, y_pred)))
 


print("Best MLP Params : ", clf.best_params_)

df_2017 = pd.read_csv("2017.csv")
y_2017 = df_2017.iloc[:,3].values
features_2017 = scaler.fit_transform(df_2017.iloc[:,5:])
 

"""
We use a MLP with 2 hidden layer 
"""
 
print("~~~~~~~~2017~~~~~~~~")

 

### Splitting Combined data

X_2017_train, X_2017_test, y_2017_train, y_2017_test = train_test_split(features_2017, y_2017, test_size=0.33, random_state=42)


# Set the parameters by cross-validation
tuned_parameters = [{'hidden_layer_sizes':[(100, 2),(1, 2), (10, 2),(30,2),(50,2) ],
                     'max_iter':[1000],
                     'activation' : ['identity', 'tanh', 'relu'],
                     'alpha' : [0.0001,0.02],
                     'solver' : [   'sgd', 'adam'],
                     }]
 
clf = GridSearchCV(MLPRegressor(), tuned_parameters)
clf.fit(X_2017_train, y_2017_train)
print(clf.best_params_)
y_true, y_pred = y_2017_test, clf.predict(X_2017_test)
print("Mean squared error MLP for 2017: %.2f "% sqrt(mean_squared_error(y_true, y_pred)))

print("Best MLP Params : ", clf.best_params_)

print("~~~~~~~~Combined~~~~~~~~")

"""
We combine dataset of all 3 years and perform similar analysis.
"""

features_combined = np.concatenate((features_2015,features_2016,features_2017), axis=0)


y_combined = np.concatenate((y_2015, y_2016, y_2017), axis=0)

### Splitting Combined data

X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(features_combined, y_combined, test_size=0.33, random_state=42)


# Set the parameters by cross-validation
tuned_parameters = [{'hidden_layer_sizes':[(100, 2),(1, 2), (10, 2),(30,2),(50,2) ],
                     'max_iter':[1000],
                     'activation' : ['identity', 'tanh', 'relu'],
                     'alpha' : [0.0001,0.02],
                     'solver' : [  'sgd', 'adam'],
                     }]


    
clf = GridSearchCV(MLPRegressor(), tuned_parameters)
clf.fit(X_combined_train, y_combined_train)
print(clf.best_params_)
y_true, y_pred = y_combined_test, clf.predict(X_combined_test)
print("Mean squared error MLP: %.2f"% sqrt(mean_squared_error(y_true, y_pred)))
  
 
model =MLPRegressor(activation= 'identity', alpha= 0.02, hidden_layer_sizes= (10, 2), max_iter= 1000).fit(X_combined_train, y_combined_train)
pd.DataFrame(model.loss_curve_).plot()


"""
We drop non-common feature columns for 2015 and 2016 dataset.
"""

df_2015 = df_2015.drop(columns=['Standard Error','Region','Happiness Rank'])
df_2016 = df_2016.drop(columns=['Happiness Rank','Lower Confidence Interval','Upper Confidence Interval','Region'])

"""
We drop non-common feature columns.
2017 dataset contains Column Names which are named different.
We rename those columns as per 2015 and 2016 dataset.
"""
df_2017 = df_2017.drop(columns=['Happiness.Rank','Whisker.high','Whisker.low'])
df_2017.rename(columns={'Economy..GDP.per.Capita.':'Economy (GDP per Capita)',
                          'Health..Life.Expectancy.':'Health (Life Expectancy)',
                          'Trust..Government.Corruption.':'Trust (Government Corruption)',
                          'Dystopia.Residual':'Dystopia Residual',
                          'Happiness.Score':'Happiness Score',
                          'Happiness.Rank':'Happiness Rank'}, inplace=True)

"""
For Plotting Correlation Matrix
"""
sns.set(style="whitegrid", palette="muted")
current_palette = sns.color_palette()

df_combined = pd.concat([df_2015, df_2016, df_2017], sort=True)

corrmat = df_combined.corr()
sns.heatmap(corrmat, vmax=.9, square=True)


"""
Plotting 2 features v/s Happiness score.
"""
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(221, projection='3d')
f1 = df_2015['Economy (GDP per Capita)'].values
f2 = df_2015['Family'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Economy (GDP per Capita)')
ax.set_ylabel('Family')
ax.set_zlabel('Happiness Score')

ax = fig.add_subplot(222, projection='3d')
f1 = df_2015['Trust (Government Corruption)'].values
f2 = df_2015['Freedom'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Trust (Government Corruption)')
ax.set_ylabel('Freedom')
ax.set_zlabel('Happiness Score')


ax = fig.add_subplot(223, projection='3d')
f1 = df_2015['Health (Life Expectancy)'].values
f2 = df_2015['Family'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Health (Life Expectancy)')
ax.set_ylabel('Family')
ax.set_zlabel('Happiness Score')


ax = fig.add_subplot(224, projection='3d')
f1 = df_2015['Generosity'].values
f2 = df_2015['Dystopia Residual'].values
y = df_2015['Happiness Score'].values
ax.scatter(f1, f2, y)
ax.set_xlabel('Generosity')
ax.set_ylabel('Dystopia Residual')
ax.set_zlabel('Happiness Score')

 



