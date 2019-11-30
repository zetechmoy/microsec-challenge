
#https://github.com/Vibhuti-B/NILM/blob/master/model.ipynb

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm,tree, ensemble,model_selection
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

################################################################################
#Load data

#TimestampSec  MaxCurrent  EffCurrent  Time Period  Standalone_Time  PC Screen1
data_pc1_df = pd.read_csv("data/data_pc1.csv")
data_pc1_df.drop('TimestampSec',axis=1,inplace=True)
data_pc1_df.drop('Standalone_Time',axis=1,inplace=True)
data_pc1 = np.asarray(data_pc1_df)

#TimestampSec  MaxCurrent  EffCurrent  Time Period  Standalone_Time  PC Screen2
data_pc2_df = pd.read_csv("data/data_pc2.csv")
data_pc2_df.drop('TimestampSec',axis=1,inplace=True)
data_pc2_df.drop('Standalone_Time',axis=1,inplace=True)
data_pc2 = np.asarray(data_pc2_df)

#TimestampSec  MaxCurrent  EffCurrent  Time Period  Temperature Sensor
data_sensor_df = pd.read_csv("data/data_sensor.csv")
data_sensor_df.drop('TimestampSec',axis=1,inplace=True)
data_sensor = np.asarray(data_sensor_df)

#TimestampSec  MaxCurrent  EffCurrent  Time Period
data_main_df = pd.read_csv("data/data_main.csv")
data_main_df.drop('TimestampSec',axis=1,inplace=True)
data_main = np.asarray(data_main_df)

################################################################################
#Shuffle data
data_pc1 = shuffle(data_pc1)
data_pc2 = shuffle(data_pc2)
data_sensor = shuffle(data_sensor)
data_main = shuffle(data_main)

#print(data_pc1[0:3])
#print(data_pc2[0:3])
#print(data_sensor[0:3])
#print(data_main[0:3])

################################################################################
#Prepare X and Y => divide in train/test
train_pc1, test_pc1 = train_test_split(data_pc1, test_size=0.2)
train_pc2, test_pc2 = train_test_split(data_pc2, test_size=0.2)
train_sensor, test_sensor = train_test_split(data_sensor, test_size=0.2)

x_train_pc1, y_train_pc1 = train_pc1[:,0:3], train_pc1[:,3:4]
x_test_pc1, y_test_pc1 = test_pc1[:,0:3], test_pc1[:,3:4]

x_train_pc2, y_train_pc2 = train_pc2[:,0:3], train_pc2[:,3:4]
x_test_pc2, y_test_pc2= test_pc2[:,0:3], test_pc2[:,3:4]

x_train_sensor, y_train_sensor = train_sensor[:,0:3], train_sensor[:,3:4]
x_test_sensor, y_test_sensor= test_sensor[:,0:3], test_sensor[:,3:4]
#y_train_sensor = np.reshape(y_train_sensor, (y_train_sensor.shape[0],))
#y_test_sensor = np.reshape(y_test_sensor, (y_test_sensor.shape[0],))

################################################################################
#Define models

pc1_model = ensemble.RandomForestClassifier(max_features = 3, oob_score = True, random_state = 42)
pc2_model = ensemble.RandomForestClassifier(max_features = 3, oob_score = True, random_state = 42)
sensor_model = linear_model.LogisticRegression(C=100, random_state = 42)

################################################################################
#learn !!
pc1_model.fit(x_train_pc1, y_train_pc1)
pc2_model.fit(x_train_pc2, y_train_pc2)
sensor_model.fit(x_train_sensor, y_train_sensor)

################################################################################
#evaluate and show some output
y_pred_pc1 = pc1_model.predict(x_test_pc1)
y_pred_pc2 = pc2_model.predict(x_test_pc2)
y_pred_sensor = sensor_model.predict(x_test_sensor)

print(y_pred_pc1)
print(y_pred_pc2)
print(y_pred_sensor)

print(metrics.accuracy_score(y_test_pc1,y_pred_pc1))
print(metrics.accuracy_score(y_test_pc2,y_pred_pc2))
print(metrics.accuracy_score(y_test_sensor,y_pred_sensor))

################################################################################
#save models
output = open('models/pc1_model.pkl', 'wb')
pickle.dump(pc1_model, output)
output.close()

output = open('models/pc2_model.pkl', 'wb')
pickle.dump(pc2_model, output)
output.close()

output = open('models/sensor_model.pkl', 'wb')
pickle.dump(sensor_model, output)
output.close()
