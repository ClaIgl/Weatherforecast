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
from tensorflow import keras


data2020 = pd.read_csv('C:/Users/thoma/Desktop/Semester_2/deep_LEARNIG/project/ugz_ogd_meteo_h1_2020.csv')
data2019 = pd.read_csv('C:/Users/thoma/Desktop/Semester_2/deep_LEARNIG/project/ugz_ogd_meteo_h1_2019.csv')
data2018 = pd.read_csv('C:/Users/thoma/Desktop/Semester_2/deep_LEARNIG/project/ugz_ogd_meteo_h1_2018.csv')
data2017 = pd.read_csv('C:/Users/thoma/Desktop/Semester_2/deep_LEARNIG/project/ugz_ogd_meteo_h1_2017.csv')


#%%

# subset data to one location and extract the parameters and values
ds2020 = data2020.loc[data2020['Standort'] == 'Zch_Stampfenbachstrasse', :]
dp2020 = ds2020.pivot_table(values='Wert', index='Datum', columns=['Parameter', 'Einheit'])
ds2019 = data2019.loc[data2019['Standort'] == 'Zch_Stampfenbachstrasse', :]
dp2019 = ds2019.pivot_table(values='Wert', index='Datum', columns=['Parameter', 'Einheit'])
ds2018 = data2018.loc[data2018['Standort'] == 'Zch_Stampfenbachstrasse', :]
dp2018 = ds2018.pivot_table(values='Wert', index='Datum', columns=['Parameter', 'Einheit'])
ds2017= data2017.loc[data2017['Standort'] == 'Zch_Stampfenbachstrasse', :]
dp2017 = ds2017.pivot_table(values='Wert', index='Datum', columns=['Parameter', 'Einheit'])

frames = [dp2017,dp2018,dp2019,dp2020]
dp = pd.concat(frames)


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
dp['date'] = pd.to_datetime(dp.index)
dp['month'] = dp['date'].dt.month
dp['month'] = dp['month'].astype(int)
dp.pop('date')

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

#%% remove columns
dp.pop('WVs (m/s)')
dp.pop('WVv (m/s)')
dp.pop('WD (°)')

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

#%% was selected befor normalisation

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

#%% here change to single_pred = False does not work here 
past_history = 10
future_target = 12
step = 1

data_train, labels_train = create_dataset(train_data, train_data['T (°C)'],
                                          past_history,future_target,step,single_pred=False)
data_test, labels_test = create_dataset(test_data, test_data['T (°C)'],
                                        past_history,future_target,step,single_pred=False)
data_val, labels_val = create_dataset(val_data, val_data['T (°C)'],
                                      past_history,future_target,step,single_pred=False)

#%%


single_step_model = tf.keras.models.Sequential()
single_step_model.add(keras.layers.Bidirectional(
    keras.layers.LSTM(units=128,
                      input_shape=(data_train.shape[1],data_train.shape[2]),
                                           return_sequences=True)))
single_step_model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu')))
single_step_model.add(keras.layers.Dense(units=1))

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
#%% DO NOT RUN 
'''for x, y in zip(data_val, labels_val):
    multi_step_plot(x[0], y, single_step_model.predict(np.expand_dims(x, axis=0))[0]) '''
#%%    


data_pred = single_step_model.predict(data_test)
labels_test_plot = (labels_test * train_std['T (°C)']) + train_mean['T (°C)']
data_pred_plot = (data_pred * train_std['T (°C)']) + train_mean['T (°C)']

plt.plot(labels_test_plot,'-',label='True Future')
plt.plot(data_pred_plot,'+',label='Predicted Future')
plt.legend(loc='upper left')
  
    

    
  
    
  
    
  
    
  
