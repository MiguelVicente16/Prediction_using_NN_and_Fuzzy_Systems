# Project 1 - Prediction using NN and Fuzzy Systems

# Grupo 29
# 96248	- Joao Miguel Custodio Gomes
# 96288	- Miguel Mendes Vicente

#%% 
# Importar packages necessarias
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
# Packages para implementar Fuzzy
import sklearn
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# Packages para graficos 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Packages para dividir dataset em train/test sets 
from sklearn.model_selection import train_test_split
# Packages para confusion matrix e performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
columnsLight = ['S1Light', 'S2Light', 'S3Light']

preProcessingData(df1)
normalizeData(df1)

df1.drop(columns=['Outlier'], inplace=True)

#Criacao de novas features
df1['LightSum'] = df1['S1Light'] + df1['S2Light'] + df1['S3Light']
df1['LightSum'] = df1['LightSum']/3

#Deslocar a coluna de Co2 90 minutos para tras, pq quando alguem entra demora a variar
df1['CO2Var'] = df1['CO2']
df1['CO2Var'] = df1['CO2Var'].diff(180)

#Linhas sem nada colocar a 0
for iter in range(0, 180):
      df1.iloc[iter, 13] = 0

#Comparacao Sensor perto da janela com os outros 2
df1['Luzes'] = df1['S2Light']+df1['S1Light'] 
df1['Luzes'] =  df1['S3Light']- df1['Luzes'] 

#Dividir a coluna de pessoas por 3, para ser mais facil visualizar nos graficos, dado todas as outras variaveis estarem normalizadas entre 0 e 1
df1['Persons'] = df1['Persons']/3

#Criacao da variavel weather. Explicada em pormenor no relatorio
df1['Weather'] =  np.where(((df1['Luzes']>0) & (df1['CO2Var']>0)) | ((df1['Luzes']<-1) & (df1['CO2Var']>0)),1,0)


#split train and test data
data_train, data_test = train_test_split(df1, test_size= 0.2, shuffle=False)


#Training Set
Light = data_train['LightSum']
Co2 = data_train['CO2Var']
Weather = data_train['Weather']
Persons = data_train['Persons']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Light, Co2,Persons)
ax.set_xlabel("LightSum")
ax.set_ylabel("Co2-90minA")
ax.set_zlabel("Persons")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter( Co2, Light,Persons)
ax.set_xlabel("Co2-90minA")
ax.set_ylabel("LightSum")
ax.set_zlabel("Persons")
plt.show()

# ********* IMPLEMENTACAO DO FUZZY *********

# Definir termos linguisticos e  Determinar Fuzzy Sets

Co2Var_range=np.arange(-0.55,0.55,0.01)
LightSum_range=np.arange(0,1,0.01)
Weather_range=np.arange(0,1,0.01)
Persons_range=np.arange(0,1,0.05)

Co2Var_decreasing_a_lot = fuzz.trapmf(Co2Var_range, [-0.55, -0.55, -0.45, -0.35])
Co2Var_decreasing_a_little = fuzz.trapmf(Co2Var_range, [-0.45, -0.35, -0.20,-0.10])
Co2Var_constant = fuzz.trapmf(Co2Var_range, [-0.20, -0.10, 0.05, 0.15])
Co2Var_increasing_a_little = fuzz.trapmf(Co2Var_range, [0.05, 0.15, 0.30,0.40])
Co2Var_increasing_a_lot = fuzz.trapmf(Co2Var_range, [0.30, 0.40, 0.55,0.55])

LightSum_very_low = fuzz.trapmf(LightSum_range, [0, 0, 0.15,0.25])
LightSum_low = fuzz.trapmf(LightSum_range, [0.15, 0.25, 0.35,0.45])
LightSum_medium = fuzz.trapmf(LightSum_range, [0.35, 0.45, 0.55,0.65])
LightSum_high = fuzz.trapmf(LightSum_range, [0.55, 0.65, 0.75,0.85])
LightSum_very_high = fuzz.trapmf(LightSum_range, [0.75, 0.85,1,1])

Weather_bad = fuzz.trapmf(Weather_range, [0, 0, 0.4,0.6])
Weather_good = fuzz.trapmf(Weather_range, [0.4, 0.6, 1,1])


persons_no = fuzz.trapmf(Persons_range, [0, 0, 0.4,0.6])
persons_yes = fuzz.trapmf(Persons_range, [0.4, 0.6, 1,1])

# Graficos dos Fuzzy sets
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(15, 15))

ax0.plot(Co2Var_range, Co2Var_decreasing_a_lot, 'b', linewidth=1.5, label='Decreasing a lot')
ax0.plot(Co2Var_range, Co2Var_decreasing_a_little, 'g', linewidth=1.5, label='Decreasing a little')
ax0.plot(Co2Var_range, Co2Var_constant, 'r', linewidth=1.5, label='Constant')
ax0.plot(Co2Var_range, Co2Var_increasing_a_little, 'm', linewidth=1.5, label='Increasing a little')
ax0.plot(Co2Var_range, Co2Var_increasing_a_lot, 'k', linewidth=1.5, label='Increasing a lot')
ax0.set_title('Co2 variation in a hour and half')
ax0.legend()

ax1.plot(LightSum_range, LightSum_very_low, 'b', linewidth=1.5, label='Very Low')
ax1.plot(LightSum_range, LightSum_low, 'g', linewidth=1.5, label='Low')
ax1.plot(LightSum_range, LightSum_medium, 'r', linewidth=1.5, label='Medium')
ax1.plot(LightSum_range, LightSum_high, 'm', linewidth=1.5, label='High')
ax1.plot(LightSum_range, LightSum_very_high, 'k', linewidth=1.5, label='Very High')
ax1.set_title('Light Sum')
ax1.legend()

ax2.plot(Weather_range, Weather_bad, 'r', linewidth=1.5, label='bad')
ax2.plot(Weather_range, Weather_good, 'k', linewidth=1.5, label='good')
ax2.set_title('Weather')
ax2.legend()

ax3.plot(Persons_range, persons_no, 'b', linewidth=1.5, label='No')
ax3.plot(Persons_range, persons_yes, 'g', linewidth=1.5, label='Yes')
ax3.set_title('Exceed number of persons')
ax3.legend()

plt.show()

# Definir antecedentes e consequencias
Co2Var_ant = ctrl.Antecedent(Co2Var_range,'Var of Co2')
LightSum_ant = ctrl.Antecedent(LightSum_range,'LightSum')
Weather_ant = ctrl.Antecedent(Weather_range, 'Weather')
Persons_cons = ctrl.Consequent(Persons_range,'Excess Persons (Yes/No)')

Co2Var_ant.automf(names=['Decreasing a lot','Decreasing a little','Constant','Increasing a little','Increasing a lot'])
LightSum_ant.automf(names=['Very Low','Low','Medium','High','Very High'])
Weather_ant.automf(names=['Bad','Good'])
Persons_cons.automf(names=['No','Yes'])

# Definir regras (If...Then...)
yes_case = ctrl.Rule(antecedent=(LightSum_ant['High'])|
                            (Co2Var_ant['Constant'] & LightSum_ant['Very Low']& Weather_ant['Good'])|
                            (Co2Var_ant['Constant'] & LightSum_ant['Low']& Weather_ant['Good'])|
                            (Co2Var_ant['Constant'] & LightSum_ant['Medium']& Weather_ant['Good'])|
                            (LightSum_ant['Very High'])|
                            (Co2Var_ant['Increasing a little']) |
                            (Co2Var_ant['Increasing a lot']),
                            consequent=Persons_cons['Yes'], label='Excess of Persons'
                            )

no_case = ctrl.Rule(antecedent= (Co2Var_ant['Decreasing a lot'])|
                            (Co2Var_ant['Decreasing a little'])|
                            (Co2Var_ant['Constant'] & LightSum_ant['Very Low']& Weather_ant['Bad'])|
                            (Co2Var_ant['Constant'] & LightSum_ant['Low']& Weather_ant['Bad'])|
                            (Co2Var_ant['Constant'] & LightSum_ant['Medium']& Weather_ant['Bad']),                           
                            consequent=Persons_cons['No'], label='No Excess of Persons'
                            )

control_system = ctrl.ControlSystem([no_case, yes_case])
system = ctrl.ControlSystemSimulation(control_system)

#Testing set 
Light_test = data_test['LightSum']
Co2_test = data_test['CO2Var']
Weather_test = data_test['Weather']
Persons_test = data_test['Persons']


# Aplica o Fuzzy ao test set
output = [0]*len(Persons_test)
for i in range(len(Persons_test)):
    system.input['Var of Co2'] = Co2_test[i+8105]
    system.input['LightSum'] = Light_test[i+8105]
    system.input['Weather'] = Weather_test[i+8105]
    system.compute()
    output[i] = system.output['Excess Persons (Yes/No)']

Persons_test = np.where(Persons_test==1, 1, 0)


# Graficos para estudar resultado
fig = plt.figure()
plt.plot(output,Persons_test,'o')
plt.xlabel("Fuzzy Output")
plt.ylabel("Excess Persons")
plt.show()


# Defuzzying 
for i in range(len(output)):
    if(output[i]>0.5):
        output[i]=1
    else:
        output[i]=0

cm=confusion_matrix(Persons_test,output)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Calculo da performance
# Precision
print('Precision = ',precision_score(Persons_test,output))
# Recall
print('Recall =',recall_score(Persons_test,output))
# F1 Score
print('F1 =',f1_score(Persons_test,output))


