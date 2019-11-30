
#https://github.com/Vibhuti-B/NILM/blob/master/model.ipynb

import pandas as pd
import numpy as np
import pickle
#import seaborn as sb
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

################################################################################
#Load data

#TimestampSec  MaxCurrent  EffCurrent  Time Period
data_main_df = pd.read_csv("data/data_main.csv")
data_main_df.drop('TimestampSec',axis=1,inplace=True)
data_main = np.asarray(data_main_df)

################################################################################
#Shuffle data
data_main = shuffle(data_main)
print(data_main[0:3])

################################################################################
#One hot encode every output
pkl_file = open('ohes/pc1_ohe.pkl', 'rb')
pc1_ohe = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('ohes/pc2_ohe.pkl', 'rb')
pc2_ohe = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('ohes/sensor_ohe.pkl', 'rb')
sensor_ohe = pickle.load(pkl_file)
pkl_file.close()

################################################################################
#load models
pc1_model = load_model("models/pc1_model.h5")
pc2_model = load_model("models/pc2_model.h5")
sensor_model = load_model("models/sensor_model.h5")

################################################################################
#predict
y_pred_pc1 = pc1_model.predict(data_main)
y_pred_pc2 = pc2_model.predict(data_main)
y_pred_sensor = sensor_model.predict(data_main)
################################################################################
#evaluate

data_main_df["PC screen1"] = ''
data_main_df["PC screen2"] = ''
data_main_df["Temperature sensor"] = ''

pc_labels = {0: "OFF", 1: "ON", 2: "IDLE"}
sensor_labels = {0: "OFF", 1: "ON"}

for index, row in data_main_df.iterrows():
	if index > 100:
		break
	y_pc1_out = np.zeros(y_pred_pc1[index].shape[0])
	y_pc2_out = np.zeros(y_pred_pc2[index].shape[0])
	y_sensor_out = np.zeros(y_pred_sensor[index].shape[0])

	y_pc1_out[np.argmax(y_pred_pc1[index])] = 1
	y_pc2_out[np.argmax(y_pred_pc2[index])] = 1
	y_sensor_out[np.argmax(y_pred_sensor[index])] = 1

	pc1_label = pc1_ohe.inverse_transform([y_pc1_out])[0][0]
	pc2_label = pc2_ohe.inverse_transform([y_pc2_out])[0][0]
	sensor_label = sensor_ohe.inverse_transform([y_sensor_out])[0][0]

	data_main_df.loc[index, "PC screen1"] = pc_labels[int(pc1_label)]
	data_main_df.loc[index, "PC screen2"] = pc_labels[int(pc2_label)]
	data_main_df.loc[index, "Temperature sensor"] = sensor_labels[int(sensor_label)]

data_main_df.to_csv("output.csv", index=False)
print(data_main_df.head())
