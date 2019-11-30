
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

################################################################################
#load models

pkl_file = open('models/pc1_model.pkl', 'rb')
pc1_model = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('models/pc2_model.pkl', 'rb')
pc2_model = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('models/sensor_model.pkl', 'rb')
sensor_model = pickle.load(pkl_file)
pkl_file.close()

################################################################################
#predict
y_pred_pc1 = pc1_model.predict(data_main)
y_pred_pc2 = pc2_model.predict(data_main)
y_pred_sensor = sensor_model.predict(data_main)

################################################################################
#output result

data_main_df["PC screen1"] = ''
data_main_df["PC screen2"] = ''
data_main_df["Temperature sensor"] = ''

pc_labels = {0: "OFF", 1: "ON", 2: "IDLE"}
sensor_labels = {0: "OFF", 1: "ON"}

for index, row in data_main_df.iterrows():
	print(index, end="\r")
	data_main_df.loc[index, "PC screen1"] = pc_labels[int(y_pred_pc1[index])]
	data_main_df.loc[index, "PC screen2"] = pc_labels[int(y_pred_pc2[index])]
	data_main_df.loc[index, "Temperature sensor"] = sensor_labels[int(y_pred_sensor[index])]

data_main_df.to_csv("output.csv", index=False)
print(data_main_df.head())
