
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

from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers.core import Layer, Dense, Activation, Dropout
from keras import regularizers, optimizers

def get_pc_model(X, Y):
	model = Sequential()
	model.add(Dense(16, input_shape=(X.shape[1],)))
	model.add(Dense(16, activation="relu"))
	model.add(Dense(8, activation="relu"))
	model.add(Dense(Y.shape[1], activation = 'sigmoid'))
	model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics=['accuracy'])
	return model

def get_sensor_model(X, Y):
	model = Sequential()
	model.add(Dense(4, input_shape=(X.shape[1],)))
	model.add(Dense(4, activation="sigmoid"))
	model.add(Dense(4, activation="sigmoid"))
	model.add(Dense(Y.shape[1], activation = 'sigmoid'))
	model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics=['accuracy'])
	return model

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
#One hot encode every output
pc1_ohe = OneHotEncoder(sparse=False, categories='auto')
y_train_pc1 = pc1_ohe.fit_transform(y_train_pc1)
y_test_pc1 = pc1_ohe.transform(y_test_pc1)

pc2_ohe = OneHotEncoder(sparse=False, categories='auto')
y_train_pc2 = pc2_ohe.fit_transform(y_train_pc2)
y_test_pc2 = pc2_ohe.transform(y_test_pc2)

sensor_ohe = OneHotEncoder(sparse=False, categories='auto')
y_train_sensor = sensor_ohe.fit_transform(y_train_sensor)
y_test_sensor = sensor_ohe.transform(y_test_sensor)

#save encoders
output = open('ohes/pc1_ohe.pkl', 'wb')
pickle.dump(pc1_ohe, output)
output.close()

output = open('ohes/pc2_ohe.pkl', 'wb')
pickle.dump(pc2_ohe, output)
output.close()

output = open('ohes/sensor_ohe.pkl', 'wb')
pickle.dump(sensor_ohe, output)
output.close()

################################################################################
#Define models
pc1_model = get_pc_model(x_train_pc1, y_train_pc1)
pc2_model = get_pc_model(x_train_pc2, y_train_pc2)
sensor_model = get_sensor_model(x_train_sensor, y_train_sensor)


################################################################################
#learn !!
callbacks = [
	ReduceLROnPlateau(patience=3, factor=0.1),
	EarlyStopping(patience=3)
]
pc1_model.fit(x_train_pc1, y_train_pc1, epochs=1000, batch_size= 2, validation_split=0.1, callbacks=callbacks)
pc2_model.fit(x_train_pc2, y_train_pc2, epochs=1000, batch_size= 4, validation_split=0.1, callbacks=callbacks)
sensor_model.fit(x_train_sensor, y_train_sensor, epochs=300, batch_size= 2, validation_split=0.1, callbacks=callbacks)

#pc1_model.save("pc1_model.h5")
#pc2_model.save("pc2_model.h5")
#sensor_model.save("sensor_model.h5")

################################################################################
#predict
y_pred_pc1 = pc1_model.predict(x_test_pc1)
y_pred_pc2 = pc2_model.predict(x_test_pc2)
y_pred_sensor = sensor_model.predict(x_test_sensor)

################################################################################
#evaluate and show some output
pc1_model_eval = pc1_model.evaluate(x_test_pc1, y_test_pc1)
print("pc1_model", pc1_model.metrics_names[1], pc1_model_eval[1])

pc2_model_eval = pc2_model.evaluate(x_test_pc2, y_test_pc2)
print("pc2_model", pc2_model.metrics_names[1], pc2_model_eval[1])

sensor_model_eval = sensor_model.evaluate(x_test_sensor, y_test_sensor)
print("sensor_model", sensor_model.metrics_names[1], sensor_model_eval[1])

for i in range(0, 5):
	y_pc1_out = np.zeros(y_pred_pc1[i].shape[0])
	y_pc2_out = np.zeros(y_pred_pc2[i].shape[0])
	y_sensor_out = np.zeros(y_pred_sensor[i].shape[0])

	y_pc1_out[np.argmax(y_pred_pc1[i])] = 1
	y_pc2_out[np.argmax(y_pred_pc2[i])] = 1
	y_sensor_out[np.argmax(y_pred_sensor[i])] = 1

	pc1_label = pc1_ohe.inverse_transform([y_pc1_out])[0][0]
	pc2_label = pc2_ohe.inverse_transform([y_pc2_out])[0][0]
	sensor_label = sensor_ohe.inverse_transform([y_sensor_out])[0][0]

	print(y_pc1_out, y_test_pc1[i], pc1_label)
	print(y_pc2_out, y_test_pc2[i], pc2_label)
	print(y_sensor_out, y_test_sensor[i], sensor_label)
