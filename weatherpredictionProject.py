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

features_considered = ['T (°C)', 'p (hPa)', 'Hr (%Hr)', 'RainDur (min)']

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
              yindices = range(i, i+pred_time, time_step)
              ys.append(y[yindices])

    return np.array(xs), np.array(ys)

'''def create_dataset(x, y, time_step =1):
    xs = []
    ys = []
    for i in range(len(x)-time_step):
        v = x.iloc[i:(i+time_step)].to_numpy()
        xs.append(v)
        ys.append(y.iloc[i])
    return np.array(xs), np.array(ys)'''
        
past_history = 1
future_target = 12
step = 1

data_train, labels_train = create_dataset(train_data, train_data['T (°C)'],
                                          past_history,future_target,step,single_pred=True)
data_test, labels_test = create_dataset(test_data, test_data['T (°C)'],
                                        past_history,future_target,step,single_pred=True)
data_val, labels_val = create_dataset(val_data, val_data['T (°C)'],
                                      past_history,future_target,step,single_pred=True)

#%%


single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=(data_train.shape[1], 
                                                        data_train.shape[2]),
                                           return_sequences=True))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')



# TRAIN
single_step_history = single_step_model.fit(data_train, labels_train, epochs=10,
                                            batch_size=30,
                                            validation_data=(data_test, labels_test))
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
    
plot_train_history(single_step_history, 'Loss') 
#%%
STEP = 1
def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(history), 0))
    try:
        num_out = len(true_future)
    except:
        num_out = 1
    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
               label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show() 
#%%    
for x, y in zip(data_val, labels_val):
    multi_step_plot(x[0], y, single_step_model.predict(np.expand_dims(x, axis=0))[0]) 
  
    
  
    
  
    
  
    
  
    
  
