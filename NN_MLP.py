# Project 1 - Prediction using NN and Fuzzy Systems

# Grupo 29
# 96248	- Joao Miguel Custodio Gomes
# 96288	- Miguel Mendes Vicente

#%%

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

HIDDEN_LAYER_SIZES = [7,5]
SOLVER = 'adam'
MAX_ITER = 1000
RADOM_STATE = 42
ACTIVATION_FUNC = 'relu'


def balanceData(data):

    #Converting to a data time object 
    df1["Date"] = df1["Date"] + ' ' + df1["Time"]
    data["Date"]= pd.to_datetime(data["Date"], dayfirst=True)

    #Remove the rows when there were a lot of time with no one in the room
    data.drop(data.loc[data['Date'].dt.hour>20].index, inplace=True)
    data.drop(data.loc[(data['Date'].dt.hour<8) & (data['Date'].dt.hour>1)].index, inplace=True)

    return

def normalizeData(data):
    #Min-Max Normalization
    columns = ['S1Temp', 'S2Temp', 'S3Temp', 'S1Light', 'S2Light', 'S3Light', 'CO2']
    for column in columns:
        data[column] = (data[column] - data[column].min())/(data[column].max() - data[column].min())
    return

def preProcessingData(data):

    #Two rows that escape from the outlier removal
    data.drop(labels=[5878, 3511], axis=0, inplace=True)
    
    #Columnss that can be outliers(Date and Time are not considerated)
    columns = ['S1Temp', 'S2Temp', 'S3Temp', 'S1Light', 'S2Light', 'S3Light', 'CO2', 'PIR1', 'PIR2']
    k = 6
    for column in columns:
        mean = np.mean(df1[column])
        std = np.std(df1[column])
        df1['Outlier'] = np.where(np.abs(df1[column] - mean) > np.abs(k * std), 1, 0)
        data.drop(data.loc[data['Outlier'] == 1].index, inplace=True)

    #Removing null values
    for column in data:
        index = np.where(data[column].isnull())
        data.drop(data.iloc[index].index, inplace=True)
    
    return
    
df = pd.read_csv('Proj1_Dataset.csv', sep=',', index_col=None)
df1 = df.copy()

preProcessingData(df1)
balanceData(df1)
normalizeData(df1)

df1.drop(columns=['Outlier'], inplace=True)

#df1['CO2'] = df1['CO2'].diff(100)

df1['CO2Var'] = 0
for iter in range(100, len(df1)):
      df1.iloc[iter-100, 12] = df1.iloc[iter,8]

for iter in range(len(df1)-100, len(df1)):
    df1.iloc[iter, 12] = df1.iloc[iter,8]
    
df1['CO2'] = df1['CO2Var']
df1.drop(columns=['CO2Var'], inplace=True)

X = df1.drop(axis=1,columns=['Date', 'Time', 'Persons'],inplace=False)
y = df1['Persons']

#split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#define NN
NN = MLPClassifier(solver=SOLVER, max_iter=MAX_ITER, hidden_layer_sizes=HIDDEN_LAYER_SIZES, random_state=RADOM_STATE, activation = ACTIVATION_FUNC)

#K-Fold CrossValidation
scores = cross_val_score(NN, X, y, cv=5)
print(np.mean(scores))

#train NN
NN.fit(X_train, y_train)

#make prediction from test set
y_pred = NN.predict(X_test)

#classification
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm) 
disp.plot()
plt.show()

#Classification Report
print(classification_report(y_test, y_pred))
# %%
