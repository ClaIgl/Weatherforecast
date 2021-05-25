#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:17:37 2021

@author: claraiglhaut
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import seaborn as sns

#%% FUNCTIONS

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

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.ylim(0.0,1.0)
    
    plt.show()

def multi_step_plot(history, true_future, prediction, col_no, STEP=1):
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(history), 0))
    try:
        num_out = len(true_future)
    except:
        num_out = 1
    plt.plot(num_in, history[:,col_no], label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b+',
             label='True Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r+',
               label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show() 

def multi_pred_plot(history, true_future, prediction, list_col_no, STEP=1):
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(history), 0))
    try:
        num_out = len(true_future)
    except:
        num_out = 1
    
    pred_no = len(list_col_no)
    fig, axs = plt.subplots(pred_no,1)
    
    for i in range(pred_no):
        axs[i].plot(num_in, history[:,list_col_no[i]], label='History')
        axs[i].plot(np.arange(num_out)/STEP, np.array(true_future[:,i]), 'b+',
                 label='True Future')
        if prediction.any():
          axs[i].plot(np.arange(num_out)/STEP, np.array(prediction[:,i]), 'r+',
                   label='Predicted Future')
        
    plt.legend(loc='upper left', bbox_to_anchor=(0.005, 2.8))
    plt.show()    
    
#%% LOADING DATA AND PREPROCESSING  

plt.rcParams["figure.figsize"] = (9,3)

file = '/Users/claraiglhaut/Desktop/ZHAW/Neural Networks/order95021/order_95021_data.txt'

data = pd.read_csv(file, delimiter=';')

data = data.drop('stn', axis=1)

data.columns = ['time', 'ST (°C)', 'StrGlo (W/m2)', 'p (hPa)', 'T (°C)', 
                'Snow (cm)', 'Rain (mm)', 'Hr (%)', 'SunDur (h)', 'WVs (m/s)',
                'WD (°)']

date_time_str = []
for dt in data['time']:
    date_time_str.append(str(dt)[6:8] + '.' + str(dt)[4:6] + '.' + str(dt)[0:4] + ' '  
    + str(dt)[8:10] + ':00:00')
    
data.index = pd.to_datetime(date_time_str)
data = data.drop('time', axis=1)

data = data.replace(['-'], float('NaN'))
data['Snow (cm)'] = data['Snow (cm)'].replace(float('NaN'), 0.0)

data['ST (°C)'] = data['ST (°C)'].apply(lambda x: float(x))
data['Snow (cm)'] = data['Snow (cm)'].apply(lambda x: float(x))
data['Rain (mm)'] = data['Rain (mm)'].apply(lambda x: float(x))
data['WVs (m/s)'] = data['WVs (m/s)'].apply(lambda x: float(x))
data['WD (°)'] = data['WD (°)'].apply(lambda x: float(x))

#%% look for missing data and show summary statistics
print(data.isnull().sum())

data = data.interpolate(method='time')
print(data.describe().transpose())

data['SunDur (h)'] = data['SunDur (h)'].replace(1.2, 1.0)

#%% plot features for one year 
plt.figure(1)
_ = data[0:8760].plot(subplots=True, layout=(5,2))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
plt.close()

#%% Convert wind velocity and and direction into a wind vector
plt.figure(2)
plt.hist2d(data['WD (°)'], data['WVs (m/s)'], bins=(50,50))
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
fig = plt.gcf()
fig.set_size_inches(10, 7)
plt.show()
plt.close()

wv = data['WVs (m/s)']

# Convert to radians
wd_rad = data['WD (°)']*np.pi/180.0

data['Wx (m/s)'] = wv*np.cos(wd_rad)
data['Wy (m/s)'] = wv*np.sin(wd_rad)

plt.figure(3)
plt.hist2d(data['Wx (m/s)'], data['Wy (m/s)'], bins=(50,50), vmax=50)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')
fig = plt.gcf()
fig.set_size_inches(10,7)
plt.show()
plt.close()

# drop columns we do not need 
data = data.drop(columns = ['WVs (m/s)',  'WD (°)'])

#%% Show year and day signal for Temperature
fft = tf.signal.rfft(data['T (°C)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(data['T (°C)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.ylim([0, 600000])
plt.xlim([-10, 370])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])

#%% Convert time of day and time of year signal 

time_delta = data.index - datetime.datetime(2020, 1, 1)
data['sec'] = time_delta.total_seconds()

day = 24*60*60
year = (365.2425)*day

data['Day sin'] = np.sin(data['sec'] * (2*np.pi/day))
#data['Day cos'] = np.cos(dp['sec'] * (2*np.pi/day))

data['Year sin'] = np.sin(data['sec'] * (2*np.pi/year))
#data['Year cos'] = np.cos(data['sec'] * (2*np.pi/year))

plt.figure(4)
plt.plot(np.array(data['Day sin'])[:25])
#plt.plot(np.array(data['Day cos'])[:25])

plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.show()
plt.close()

plt.figure(5)
plt.plot(np.array(data['Year sin']))
#plt.plot(np.array(dp['Year cos']))

plt.xlabel('Time [h]')
plt.title('Time of year signal')
plt.show()
plt.close()

# drop sec column
data = data.drop('sec', axis=1)

#%% feature heatmap
corrMat = data.corr()
sns.heatmap(corrMat) 
fig = plt.gcf()
fig.set_size_inches(10,7)
plt.show()
plt.close()


#%% train, test, validation split 

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

num_features = data.shape[1]

#%% normalize data with mean 

train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)

train_data = (train_data - train_mean)/ train_std
val_data = (val_data - train_mean)/ train_std
test_data = (test_data - train_mean)/ train_std

train_data.boxplot(rot=90)
fig = plt.gcf()
fig.set_size_inches(10,12)


#%% BUILDING MODELS

#%% Prediction for a single parameter T with all features

features_considered = ['ST (°C)', 'StrGlo (W/m2)', 'p (hPa)', 'T (°C)', 'Snow (cm)',
       'Rain (mm)', 'Hr (%)', 'SunDur (h)', 'Wx (m/s)', 'Wy (m/s)', 'Day sin',
       'Year sin']

features = data[features_considered]
features.plot(subplots=True)
features.head()


train_X = train_data[features_considered]
val_X = val_data[features_considered]
test_X = test_data[features_considered]

past_history = 120
future_target = 12
step = 1

X_train, y_train = create_dataset(train_X, train_X[['T (°C)']],
                                          past_history,future_target,step,single_pred = False)
X_test, y_test = create_dataset(test_X, test_X[['T (°C)']],
                                        past_history,future_target,step,single_pred = False)
X_val, y_val = create_dataset(val_X, val_X[['T (°C)']],
                                      past_history,future_target,step,single_pred = False)

# model loss: 0.1522 - val_loss: 0.1748 history=120, future target = 12
'''model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=(X_train.shape[1], 
                                                        X_train.shape[2]))))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dense(12))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')'''

# model 
'''model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=(X_train.shape[1], 
                                                        X_train.shape[2]))))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(12))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')'''

# model loss: 0.2283 - val_loss: 0.2003 history=300, future_target = 12
# model loss: 0.2273 - val_loss: 0.1963 history=120, future_target = 12
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True,
                                           input_shape=(X_train.shape[1], 
                                                        X_train.shape[2]))))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, activation='softmax')))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(12))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

# TRAIN
history = model.fit(X_train, y_train, epochs=15,
                                            batch_size=50,
                                            validation_data=(X_val, y_val))


plot_train_history(history, 'Loss') 

#%%
for x, y in list(zip(X_test, y_test))[0:3]:
    multi_step_plot(x, y, model.predict(np.expand_dims(x, axis=0))[0],3)
#%% Prediction for two Parameters 





