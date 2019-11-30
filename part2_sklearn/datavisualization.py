
#https://github.com/Vibhuti-B/NILM/blob/master/model.ipynb

import pandas as pd
import numpy as np
#import seaborn as sb
import matplotlib.pyplot as plt
#read data
data_pc1 = pd.read_csv("data/test_data1536128609.98_screen1.csv", header=None)
print("data_pc1", data_pc1.shape)
print(data_pc1.head())

data_pc2 = pd.read_csv("data/test_data1536128871.84_screen2.csv", header=None)
print("data_pc2", data_pc2.shape)
print(data_pc2.head())

data_sensor = pd.read_csv("data/test_data1536129261.77_device253.csv", header=None)
print("data_sensor", data_sensor.shape)
print(data_sensor.head())

data_main = pd.read_csv("data/test_data1536129670.55_main.csv", header=None)
print("data_main", data_main.shape)
print(data_main.head())

#rename cols
col_names = { 0:"TimestampSec", 1:"MaxCurrent", 2:"EffCurrent" }
data_pc1.rename(columns=col_names, inplace=True)
data_pc2.rename(columns=col_names, inplace=True)
data_sensor.rename(columns=col_names, inplace=True)
data_main.rename(columns=col_names, inplace=True)

plot_pc1 = data_pc1.plot.line(x='TimestampSec', y='EffCurrent',title='PC 1')
plot_pc2 = data_pc2.plot.line(x='TimestampSec', y='EffCurrent',title='PC 2')
plot_sensor = data_sensor.plot.line(x='TimestampSec', y='EffCurrent',title='Sensor')
plot_main = data_main.plot.line(x='TimestampSec', y='EffCurrent',title='Sensor')
plt.show()
