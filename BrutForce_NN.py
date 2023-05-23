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

HIDDEN_LAYER_SIZES = [4]
SOLVER = 'adam'
MAX_ITER = 300
RADOM_STATE = 1

ACTIVATION_FUNC = 'relu'

def preProcessingData(data):
    #Remove null values
    for column in data:
        index = np.where(data[column].isnull())
        data.drop(data.iloc[index].index, inplace=True)
    return
    
#Read data from file
df = pd.read_csv('Proj1_Dataset.csv', sep=',', index_col=None)
df1 = df.copy()

preProcessingData(df1)

X = df1.drop(axis=1,columns=['Date', 'Time', 'Persons'],inplace=False)
y = df1['Persons']

#split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#define NN
NN = MLPClassifier(solver=SOLVER, max_iter=MAX_ITER, hidden_layer_sizes=HIDDEN_LAYER_SIZES, random_state=RADOM_STATE)

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

print(classification_report(y_test, y_pred))

# %%
