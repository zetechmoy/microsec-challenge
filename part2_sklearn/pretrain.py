
#https://github.com/Vibhuti-B/NILM/blob/master/model.ipynb

import pandas as pd
import numpy as np
#import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm,tree, ensemble,model_selection
from sklearn.utils import shuffle

#read data
data_pc1 = pd.read_csv("data/test_data1536128609.98_screen1.csv", header=None)
print("data_pc1", data_pc1.shape)
#print(data_pc1.head())

data_pc2 = pd.read_csv("data/test_data1536128871.84_screen2.csv", header=None)
print("data_pc2", data_pc2.shape)
#print(data_pc2.head())

data_sensor = pd.read_csv("data/test_data1536129261.77_device253.csv", header=None)
print("data_sensor", data_sensor.shape)
#print(data_sensor.head())

data_main = pd.read_csv("data/test_data1536129670.55_main.csv", header=None)
print("data_main", data_main.shape)
#print(data_main.head())

#rename cols
col_names = { 0:"TimestampSec", 1:"MaxCurrent", 2:"EffCurrent" }
data_pc1.rename(columns=col_names, inplace=True)
data_pc2.rename(columns=col_names, inplace=True)
data_sensor.rename(columns=col_names, inplace=True)
data_main.rename(columns=col_names, inplace=True)

#pc1
#define standalone time and transition time
print("Computing pc1...")
data_pc1['Time Period']=''
data_pc1['Standalone_Time']=''
for index,row in data_pc1.iterrows():
    if(index==0):
        data_pc1['Standalone_Time'][0]=0
        data_pc1['Time Period'][0]=0
    elif(data_pc1['EffCurrent'][index-1]==data_pc1['EffCurrent'][index]):
        data_pc1['Standalone_Time'][index]=round(data_pc1['TimestampSec'][index]-data_pc1['TimestampSec'][index-1],2)
        data_pc1['Time Period'][index]=round(data_pc1['TimestampSec'][index]-data_pc1['TimestampSec'][index-1],2)
    else:
        data_pc1['Standalone_Time'][index]=0
        data_pc1['Time Period'][index]=round(data_pc1['TimestampSec'][index]-data_pc1['TimestampSec'][index-1],2)

data_pc1['PC Screen1']=''
data_pc1['EffCurrent_Diff'] = data_pc1['EffCurrent'].diff()
data_pc1.loc[(data_pc1['EffCurrent_Diff']== 0), 'PC Screen1'] = 2
data_pc1.loc[(data_pc1['EffCurrent_Diff']!= 0), 'PC Screen1'] = 1
data_pc1.loc[(data_pc1['EffCurrent'] == 0), 'PC Screen1'] = 0
data_pc1.drop('EffCurrent_Diff',axis=1,inplace=True)
data_pc1.to_csv("data/data_pc1.csv", index=False)

#pc2
#same thing for pc2
print("Computing pc2...")
data_pc2['Time Period']=''
data_pc2['Standalone_Time']=''
for index,row in data_pc2.iterrows():
    if(index==0):
        data_pc2['Standalone_Time'][0]=0
        data_pc2['Time Period'][0]=0
    elif(data_pc2['EffCurrent'][index-1]==data_pc2['EffCurrent'][index]):
        data_pc2['Standalone_Time'][index]=round(data_pc2['TimestampSec'][index]-data_pc2['TimestampSec'][index-1],2)
        data_pc2['Time Period'][index]=round(data_pc2['TimestampSec'][index]-data_pc2['TimestampSec'][index-1],2)
    else:
        data_pc2['Standalone_Time'][index]=0
        data_pc2['Time Period'][index]=round(data_pc2['TimestampSec'][index]-data_pc2['TimestampSec'][index-1],2)

data_pc2['PC Screen2']=''
data_pc2['EffCurrent_Diff'] = data_pc2['EffCurrent'].diff()
data_pc2.loc[(data_pc2['EffCurrent_Diff']== 0), 'PC Screen2'] = 2
data_pc2.loc[(data_pc2['EffCurrent_Diff']!= 0), 'PC Screen2'] = 1
data_pc2.loc[(data_pc2['EffCurrent'] == 0), 'PC Screen2'] = 0
data_pc2.drop('EffCurrent_Diff',axis=1,inplace=True)
data_pc2.to_csv("data/data_pc2.csv", index=False)

#sensor
#data_sensor['Time Period']=''
print("Computing sensor...")
data_sensor['Time Period']=''
for index,row in data_sensor.iterrows():
    if(index==0):
        data_sensor['Time Period'][0]=0
    else:
        data_sensor['Time Period'][index]=round(data_sensor['TimestampSec'][index]-data_sensor['TimestampSec'][index-1],2)

data_sensor['Temperature Sensor']=''
data_sensor['EffCurrent_Diff'] = data_sensor['EffCurrent'].diff()
data_sensor.loc[(data_sensor['EffCurrent_Diff']== 0), 'Temperature Sensor'] = 0
data_sensor.loc[(data_sensor['EffCurrent_Diff']!= 0), 'Temperature Sensor'] = 1
data_sensor.loc[(data_sensor['EffCurrent'] == 0), 'Temperature Sensor'] = 0
data_sensor.drop('EffCurrent_Diff',axis=1,inplace=True)
data_sensor.to_csv("data/data_sensor.csv", index=False)

#same thing for main data
print("Computing main...")
data_main['Time Period']=''
for index,row in data_main.iterrows():
    if(index==0):
        data_main['Time Period'][0]=0
    else:
        data_main['Time Period'][index]=round(data_main['TimestampSec'][index]-data_main['TimestampSec'][index-1],2)
data_main.to_csv("data/data_main.csv", index=False)

exit()

data_sensor = shuffle(data_sensor)
data_pc1 = shuffle(data_pc1)
data_pc2 = shuffle(data_pc2)

train_data,validate=train_test_split(data_pc1,test_size=0.2)
x_train_pc1 = pd.DataFrame(train_data.iloc[:,1:4])
y_train_pc1=pd.DataFrame(train_data.iloc[:,-1])
x_test_pc1 = pd.DataFrame(validate.iloc[:,1:4])
y_test_pc1=pd.DataFrame(validate.iloc[:,-1])

print(x_train_pc1.head())
print(y_train_pc1.head())

train_data,validate=train_test_split(data_pc2,test_size=0.2)
x_train_pc2 = pd.DataFrame(train_data.iloc[:,1:4])
y_train_pc2=pd.DataFrame(train_data.iloc[:,-1])
x_test_pc2 = pd.DataFrame(validate.iloc[:,1:4])
y_test_pc2=pd.DataFrame(validate.iloc[:,-1])

train_data,validate=train_test_split(data_sensor,test_size=0.2)
x_train_sensor = pd.DataFrame(train_data.iloc[:,1:4])
y_train_sensor=pd.DataFrame(train_data.iloc[:,-1])
x_test_sensor = pd.DataFrame(validate.iloc[:,1:4])
y_test_sensor=pd.DataFrame(validate.iloc[:,-1])

################################################################################

#models={'logit':'','svm':'','bagging':'','rforest':'','adaboost':'','gboost':''}
#regularization=[0.001,0.01,0.1,1,10,100,1000]
#
#print(x_train_pc1.head())
#print(y_train_pc1.head())
#x_train_pc1.to_csv("x_train_pc1.csv", index=False)
#print(x_train_pc1)
#x_train_pc1 = np.asarray(x_train_pc1, dtype=np.float64)
#y_train_pc1 = np.asarray(y_train_pc1, dtype=np.float64)
#
###Logistic Regression
#print("Logistic Regression")
#scores=[]
#for c in regularization:
#    logit=linear_model.LogisticRegression(C=c)
#    logit.fit(x_train_pc1,y_train_pc1)
#    scores.append(logit.score(x_test_pc1, y_test_pc1))
#
#c=regularization[np.argmax(scores)]
#logit=linear_model.LogisticRegression(C=c)
#logit.fit(x_train_pc1,y_train_pc1)
#print("With C= ",c," Score is:",logit.score(x_test_pc1, y_test_pc1))
#
#models['logit']=logit.score(x_test_pc1, y_test_pc1)
#y_pred=logit.predict(x_test_pc1)
#
#labels = [0,1,2]
#print("Accuracy= ",round(logit.score(x_test_pc1,y_test_pc1)*100,1),"%")
#
###Support Vector Machines
#print("Support Vector Machines")
#scores=[]
#for c in regularization:
#    svc= svm.LinearSVC(C=c)
#    svc.fit(x_train_pc1,y_train_pc1)
#    print("Accuracy for C= ",c," is= ", svc.score(x_test_pc1,y_test_pc1))
#    scores.append(svc.score(x_test_pc1,y_test_pc1))
#
#c=regularization[np.argmax(scores)]
#svc=svm.LinearSVC(C=c)
#svc.fit(x_train_pc1,y_train_pc1)
#y_pred_svm=svc.predict(x_test_pc1)
#
#print("Accuracy= ",round(svc.score(x_test_pc1,y_test_pc1)*100,1),"%")
#
#models['svm']=svc.score(x_test_pc1, y_test_pc1)
#
#
##Bagging
#print("Bagging")
#bagging = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth=20,max_features=3), max_samples = 0.5, max_features = 3, oob_score = True, random_state = 2018)
#bagging.fit(x_train_pc1, y_train_pc1)
#print("Accuracy: ",bagging.score(x_test_pc1,y_test_pc1))
#models['bagging']=bagging.score(x_test_pc1, y_test_pc1)
#
##Boosting with Random Forest
#print("Boosting with Randome Forest")
#rforest = ensemble.RandomForestClassifier(max_features = 3, oob_score = True, random_state = 2018)
#rforest.fit(x_train_pc1, y_train_pc1)
#
#print("Accuracy: ",rforest.score(x_test_pc1,y_test_pc1))
#models['rforest']=rforest.score(x_test_pc1, y_test_pc1)
#
##Adaboosting
#print("AdaBoost")
#adaboost = [
#    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME', random_state = 2018),
#    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME.R', random_state = 2018)]
#
#for i in range(2):
#    adaboost[i].fit(x_train_pc1, y_train_pc1)
#
#scores=[]
#for i in range(2):
#    scores.append(adaboost[i].score(x_test_pc1, y_test_pc1))
#    print(adaboost[i].score(x_test_pc1, y_test_pc1))
#
#models['adaboost']=np.max(scores)
#
##Gradient Boosting
#print("Gradient Boosting")
#gboost = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 2018)
#gboost.fit(x_train_pc1, y_train_pc1)
#y_pred_gboost=gboost.predict(x_test_pc1)
#print("Accuracy for Gradient Boosting: ", metrics.accuracy_score(y_test_pc1,y_pred_gboost))
#
#models['gboost']=metrics.accuracy_score(y_test_pc1,y_pred_gboost)
#
###Running the model on the test Data
#
#model=max(models,key=models.get)
#print("Best Model is : ",model)

rforest = ensemble.RandomForestClassifier(max_features = 3, oob_score = True, random_state = 2018)
rforest.fit(x_train_pc1, y_train_pc1)

#svc=svm.LinearSVC(C=10)
#svc.fit(x_train_pc2,y_train_pc2)
#
#logit=linear_model.LogisticRegression(C=100)
#logit.fit(x_train_sensor,y_train_sensor)

data_main['Time Period']=''
for index,row in data_main.iterrows():
    if(index==0):
        data_main['Time Period'][0]=0
    else:
        data_main['Time Period'][index]=round(data_main['TimestampSec'][index]-data_main['TimestampSec'][index-1],2)

y_pc1_pred=rforest.predict(pd.DataFrame(data_main.iloc[:,1:4]))
#y_pc2_pred=svc.predict(pd.DataFrame(test.iloc[:,1:4]))
#y_sensor_pred=logit.predict(pd.DataFrame(test.iloc[:,1:4]))

frame=pd.DataFrame(list(zip(y_pc1_pred)))
frame.rename(columns={0:'PC Screen1',1:'PC Screen2',2:'Temperature Sensor'},inplace=True)
data_main=pd.concat([data_main,frame],axis=1,sort=False)
data_main.head()
