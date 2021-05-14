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

#%% Convert time of day and time of year signal - year signal is weird 
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

#%%

n = len(dp)
train_dp = dp[0:int(n*0.7)]
val_dp = dp[int(n*0.7):int(n*0.9)]
test_dp = dp[int(n*0.9):]

num_features = dp.shape[1]

#%% normalize data with mean 

train_mean = train_dp.mean()
train_std = train_dp.std()

train_dp = (train_dp - train_mean)/ train_std
val_dp = (val_dp - train_mean)/ train_std
test_dp = (test_dp - train_mean)/ train_std

train_dp.boxplot(rot=90)

#%%

features_considered = ['T (°C)', 'p (hPa)', 'Hr (%Hr)', 'RainDur (min)']

features = dp[features_considered]
features.plot(subplots=True)
features.head()


features_norm = train_dp[features_considered]






