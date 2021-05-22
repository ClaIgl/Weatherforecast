#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 08:51:37 2021

@author: claraiglhaut
"""
import os
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras


data = pd.read_csv('/Users/claraiglhaut/Desktop/ZHAW/Neural Networks/Data_2020')

#%%

# subset data to one location and extract the parameters and values
ds = data.loc[data['Standort'] == 'Zch_Stampfenbachstrasse', :]
dp = ds.pivot_table(values='Wert', index='Datum', columns=['Parameter', 'Einheit'])

dp.columns = [' ('.join(column)+')' for column in dp.columns]

date_time = pd.to_datetime(dp.index)
date_time = pd.to_datetime(date_time.strftime('%Y-%m-%d %H:%M:%S'))

dp.index = date_time


#%% plot features
plt.figure(1)
_ = dp.plot(subplots=True)
plt.show()
plt.close()

#%% impute missing data
print(dp.isnull().sum())

dp = dp.interpolate(method='time')
print(dp.describe().transpose())

#%% Convert wind velocity and and direction into a wind vector
plt.figure(2)
plt.hist2d(dp['WD (°)'], dp['WVs (m/s)'], bins=(50,50))
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
plt.show()
plt.close()


#%%
wv = dp['WVs (m/s)']

# Convert to radians
wd_rad = dp['WD (°)']*np.pi/180.0

dp['Wx (m/s)'] = wv*np.cos(wd_rad)
dp['Wy (m/s)'] = wv*np.sin(wd_rad)

plt.figure(3)
plt.hist2d(dp['Wx (m/s)'], dp['Wy (m/s)'], bins=(50,50), vmax=50)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')
plt.show()
plt.close()

# drop
dp = dp.drop(columns = ['WVs (m/s)', 'WVv (m/s)', 'WD (°)'])

#%%
fft = tf.signal.rfft(dp['T (°C)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(dp['T (°C)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.ylim([0, 40000])
plt.xlim([0, 370])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])

#%% Convert time of day and time of year signal 

time_delta = dp.index - datetime.datetime(2020, 1, 1)
dp['sec'] = time_delta.total_seconds()

day = 24*60*60
year = (365.2425)*day

dp['Day sin'] = np.sin(dp['sec'] * (2*np.pi/day))
dp['Day cos'] = np.cos(dp['sec'] * (2*np.pi/day))

dp['Year sin'] = np.sin(dp['sec'] * (2*np.pi/year))
dp['Year cos'] = np.cos(dp['sec'] * (2*np.pi/year))

plt.figure(4)
plt.plot(np.array(dp['Day sin'])[:25])
plt.plot(np.array(dp['Day cos'])[:25])

plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.show()
plt.close()

plt.figure(5)
plt.plot(np.array(dp['Year sin']))
plt.plot(np.array(dp['Year cos']))

plt.xlabel('Time [h]')
plt.title('Time of year signal')
plt.show()
plt.close()

# drop sec column
dp = dp.drop('sec', axis=1)
#%% feature heatmap
corrMat = dp.corr()
sns.heatmap(corrMat) 
plt.show()

#%% train, test, validation split 

n = len(dp)
train_dp = dp[0:int(n*0.7)]
val_dp = dp[int(n*0.7):int(n*0.9)]
test_dp = dp[int(n*0.9):]

num_features = dp.shape[1]

#%% normalize data with mean 

train_mean = train_dp.mean(axis=0)
train_std = train_dp.std(axis=0)

train_dp = (train_dp - train_mean)/ train_std
val_dp = (val_dp - train_mean)/ train_std
test_dp = (test_dp - train_mean)/ train_std

train_dp.boxplot(rot=90)

#%%

features_considered = ['T (°C)', 'p (hPa)', 'Hr (%Hr)', 'RainDur (min)', 'Day sin',
                       'Year sin','Wx (m/s)','Wy (m/s)', 'StrGlo (W/m2)']

features = dp[features_considered]
features.plot(subplots=True)
features.head()


train_data = train_dp[features_considered]
val_data = val_dp[features_considered]
test_data = test_dp[features_considered]

#%% Build model

def create_dataset(x, y, history_size,
                      pred_time, time_step=1, single_pred = False):
    
    x = x.values
    y = y.values

    xs = []
    ys = []
      
    for i in range(history_size, len(x) - pred_time):
        xindices = range(i-history_size, i, time_step)
        xs.append(x[xindices])
        if single_pred:
            ys.append(y[i+pred_time])
        else:
            yindices = range(i, i+pred_time,time_step)
            ys.append(y[yindices])

    return np.array(xs), np.array(ys)

#%% Single prediction        
past_history = 300
future_target = 12
step = 1

data_train, labels_train = create_dataset(train_data, train_data[['Hr (%Hr)']],
                                          past_history,future_target,step,single_pred = False)
data_test, labels_test = create_dataset(test_data, test_data[['Hr (%Hr)']],
                                        past_history,future_target,step,single_pred = False)
data_val, labels_val = create_dataset(val_data, val_data[['Hr (%Hr)']],
                                      past_history,future_target,step,single_pred = False)

#%%

# model for two predictions
'''model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=(data_train.shape[1], 
                                                        data_train.shape[2]))))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dense(12*2))
model.add(tf.keras.layers.Reshape([12, 2]))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')'''

# model for one prediction overfits with history of 120 after epoch 5
'''model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=(data_train.shape[1], 
                                                        data_train.shape[2]))))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dense(12))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')'''

# still overfits but not as much
'''model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=(data_train.shape[1], 
                                                        data_train.shape[2]))))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(12))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')'''

# no overfitting quite highest loss (~0.4 training and ~0.35 validation) 
# (more epochs? smaller dropout? more data?)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=(data_train.shape[1], 
                                                        data_train.shape[2]))))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, activation='softmax')))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(12))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')


# TRAIN
history = model.fit(data_train, labels_train, epochs=15,
                                            batch_size=50,
                                            validation_data=(data_val, labels_val))

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    
    plt.show()
    
plot_train_history(history, 'Loss') 

#%%
modelpath = '/Users/claraiglhaut/Desktop/ZHAW/Neural Networks/model.h5'
pretrained_model = keras.models.load_model(modelpath)

history = pretrained_model.fit(data_train, labels_train, epochs=15,
                                            batch_size=50,
                                            validation_data=(data_val, labels_val))

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    
    plt.show()
    
plot_train_history(history, 'Loss') 
#%%

def multi_step_plot(history, true_future, prediction, STEP=1):
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(history), 0))
    try:
        num_out = len(true_future)
    except:
        num_out = 1
    plt.plot(num_in, history[:,0], label='History')
    plt.plot(num_in, history[:,2], label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b+',
             label='True Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r+',
               label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show() 
    
#%%  
for x, y in list(zip(data_test, labels_test))[150:153]:
    multi_step_plot(x, y, model.predict(np.expand_dims(x, axis=0))[0])
  
#%% If we want to predict two features (can be adjusted to more than two)
  
def multi_pred_plot(history, true_future, prediction, STEP=1):
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(history), 0))
    try:
        num_out = len(true_future)
    except:
        num_out = 1
    
    fig, axs = plt.subplots(2,1)
    axs[0].plot(num_in, history[:,0], label='History')
    axs[0].plot(np.arange(num_out)/STEP, np.array(true_future[:,0]), 'b+',
             label='True Future')
    if prediction.any():
      axs[0].plot(np.arange(num_out)/STEP, np.array(prediction[:,0]), 'r+',
               label='Predicted Future')
    axs[1].plot(num_in, history[:,2], label='History',c='r')
    axs[1].plot(np.arange(num_out)/STEP, np.array(true_future[:,1]), 'b+',
             label='True Future')
    if prediction.any():
      axs[1].plot(np.arange(num_out)/STEP, np.array(prediction[:,1]), 'r+',
               label='Predicted Future')
    plt.legend(loc='upper left', bbox_to_anchor=(0.005, 2.8))
    plt.show()    
  
for x, y in list(zip(data_test, labels_test))[150:153]:
    multi_pred_plot(x, y, model.predict(np.expand_dims(x, axis=0))[0])   
  
    
  
    
  
