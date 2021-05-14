# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:39:39 2021

@author: thoma
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

#from matplotlib import rc

data = pd.read_csv('ugz_ogd_meteo_h1_2020.csv')

data_sub = data.loc[data['Standort'] == 'Zch_Stampfenbachstrasse', :]

data_input = data_sub.pivot_table(values='Wert', index='Datum', columns='Parameter')

# datum in tage und monate machen


data_input.WD =  data_input.WD*np.pi / 180.0


data_input['Wx'] = data_input.WVs*np.cos(data_input.WD)
data_input['Wy'] = data_input.WVs*np.sin(data_input.WD)

data_input.pop('WD')
data_input.pop('WVs')
data_input.pop('WVv')


data_output = data_input.pop('T')



train_input = data_input[0:int(8763*0.9)]
test_input = data_input[int(8763*0.9):]

train_output = data_output[0:int(8763*0.9)]
test_output = data_output[int(8763*0.9):]


# scaling

def create_dataset(x, y, time_step =1):
    xs = []
    ys = []
    for i in range(len(x)-time_step):
        v = x.iloc[i:(i+time_step)].to_numpy()
        xs.append(v)
        ys.append(y.iloc[i + time_step])
    return np.array(xs), np.array(ys)

time_steps = 24

x_train, y_train = create_dataset(train_input, train_output, time_step = time_steps)
x_test, y_test = create_dataset(test_input, test_output, time_step = time_steps)

print(x_train.shape)

model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,input_shape=(x_train.shape[1],x_train.shape[2]))))
model.add(keras.layers.Dropout(rate=0.01))
model.add(keras.layers.Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split= 0.1,
    shuffle=False)



        