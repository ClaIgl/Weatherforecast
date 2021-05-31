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
import os

#%% FUNCTIONS

def create_dataset(x, y, history_size,
                      pred_time, time_step=1, single_pred = False):
    '''
    Creates data and label dataset with given history size, pred_time gives 
    the number of predictions.
    '''
    
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
    
    '''
    Plots training and validation loss.
    '''

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    
    plt.show()

def multi_step_plot(history, true_future, prediction, col_no, train_std, 
                    train_mean, title, STEP=1):
    
    '''
    Plots the history, the true future and the predicted future for multiple 
    steps and for one prediction paramter.
    '''
    
    plt.figure(figsize=(12, 6))
    num_in = list(range(-len(history), 0))
    try:
        num_out = len(true_future)
    except:
        num_out = 1
    plt.plot(num_in, history[:,col_no]*train_std[col_no] + train_mean[col_no], label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future)*train_std[col_no] + train_mean[col_no], 'b+',
             label='True Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/STEP, np.array(prediction)*train_std[col_no] + train_mean[col_no], 'r+',
               label='Predicted Future')
    plt.legend(loc='upper right',ncol=3)
    plt.title(title, loc= 'left')
    plt.show() 

def multi_pred_plot(history, true_future, prediction, list_col_no, train_std, 
                    train_mean, subtitles, STEP=1):
    
    '''
    Plots the history, the true future and the predicted future for multiple 
    steps and for multiple prediction paramter.
    '''
    

    num_in = list(range(-len(history), 0))
    try:
        num_out = len(true_future)
    except:
        num_out = 1
    
    pred_no = len(list_col_no)
    fig, axs = plt.subplots(pred_no, 1, sharex=True)
    fig.set_size_inches(12,10)
    for i in range(pred_no):
        axs[i].plot(num_in, history[:,list_col_no[i]]*train_std[list_col_no[i]] + train_mean[list_col_no[i]])
        axs[i].plot(np.arange(num_out)/STEP, np.array(true_future[:,i])*train_std[list_col_no[i]] 
                    + train_mean[list_col_no[i]],'+b')
        if prediction.any():
          axs[i].plot(np.arange(num_out)/STEP, np.array(prediction[:,i])*train_std[list_col_no[i]] 
                    + train_mean[list_col_no[i]],'+r')
          
        axs[i].set_title(subtitles[i], loc='left')
        
    plt.legend(['History', 'True Future', 'Predicted Future'], 
               loc='upper right', ncol=3)
    plt.show()    
    
#%% LOADING DATA AND PREPROCESSING  

plt.rcParams["figure.figsize"] = (9,5)

#file = '/Users/claraiglhaut/Desktop/ZHAW/Neural Networks/order95021/order_95021_data.txt'

file = os.path.realpath('order95021/order_95021_data.txt')
data = pd.read_csv(file, delimiter=';')

# delete location column
data = data.drop('stn', axis=1)

data.columns = ['time', 'ST (°C)', 'StrGlo (W/m2)', 'p (hPa)', 'T (°C)', 
                'Snow (cm)', 'Rain (mm)', 'Hr (%)', 'SunDur (h)', 'WVs (m/s)',
                'WD (°)']

# add datetime index and drop time column
date_time_str = []
for dt in data['time']:
    date_time_str.append(str(dt)[6:8] + '.' + str(dt)[4:6] + '.' + 
                         str(dt)[0:4] + ' ' + str(dt)[8:10] + ':00:00')
    
data.index = pd.to_datetime(date_time_str)
data = data.drop('time', axis=1)

# change missing value character to NaN and change datatype of all  columns 
# to float
data = data.replace(['-'], float('NaN'))

data['ST (°C)'] = data['ST (°C)'].apply(lambda x: float(x))
data['Snow (cm)'] = data['Snow (cm)'].apply(lambda x: float(x))
data['Rain (mm)'] = data['Rain (mm)'].apply(lambda x: float(x))
data['WVs (m/s)'] = data['WVs (m/s)'].apply(lambda x: float(x))
data['WD (°)'] = data['WD (°)'].apply(lambda x: float(x))

#%% look for and impute missing data and show summary statistics

# find missing values
print(data.isnull().sum())

# set missing values for Snow (cm) to zero
data['Snow (cm)'] = data['Snow (cm)'].replace(float('NaN'), 0.0)

# interpolate the other missing values 
data = data.interpolate(method='time')

# print summary statistic
print(data.describe().transpose())

# change unrealtistic values 
data['SunDur (h)'] = data['SunDur (h)'].replace(1.2, 1.0)

#%% plot all features for one year 

_ = data[0:8760].plot(subplots=True, layout=(5,2))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
plt.close()

#%% Convert wind velocity and and direction into a wind vector

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

# Convert to vector in x and y direction
data['Wx (m/s)'] = wv*np.cos(wd_rad)
data['Wy (m/s)'] = wv*np.sin(wd_rad)


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
plt.title('Year and Day Signal for Air Temperature')
plt.show()
plt.close()

#%% Convert time of day and time of year signal to sinus signal 

time_delta = data.index - datetime.datetime(2021, 1, 1)
data['sec'] = time_delta.total_seconds()

day = 24*60*60
year = (365.2425)*day

data['Day sin'] = np.sin(data['sec'] * (2*np.pi/day))
data['Year sin'] = np.sin(data['sec'] * (2*np.pi/year))

# plot sinus day signal
plt.plot(np.array(data['Day sin'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.show()
plt.close()

# plot sinus year signal
plt.plot(np.array(data['Year sin']))
plt.xlabel('Time [h]')
plt.title('Time of year signal')
plt.show()
plt.close()

# drop sec column
data = data.drop('sec', axis=1)

#%% visualize feature correlation 

corrMat = data.corr()
sns.heatmap(corrMat, vmax=1.0, vmin=-1.0,) 
fig = plt.gcf()
fig.set_size_inches(10,7)
plt.title('Feature Correlation Heatmap')
plt.show()
plt.close()


#%% train, validation and test split 

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

#%% normalize data with mean and standard deviation of the training set

train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)

train_data = (train_data - train_mean)/ train_std
val_data = (val_data - train_mean)/ train_std
test_data = (test_data - train_mean)/ train_std

train_data.boxplot(rot=90)
fig = plt.gcf()
fig.set_size_inches(10,12)
plt.title('Boxplot of Normalized Training Data')

#%% BUILDING THE MODEL

# features selected 
features_considered = ['ST (°C)', 'StrGlo (W/m2)', 'p (hPa)', 'T (°C)', 
                       'Snow (cm)', 'Rain (mm)', 'Hr (%)', 'SunDur (h)', 
                       'Wx (m/s)', 'Wy (m/s)', 'Day sin', 'Year sin']

train_X = train_data[features_considered]
val_X = val_data[features_considered]
test_X = test_data[features_considered]

#%% Prediction for a single parameter air temperature T (°C) with all features

#%% Hyperparameter tuning: EXAMPLE - size of first layer

parameter_optimisation = [12, 24, 32, 64]
fig = plt.figure()

for current_value in parameter_optimisation:

    past_history = 120
    future_target = 12
    step = 1

    X_train, y_train = create_dataset(
        train_X, train_X[['T (°C)']], past_history,future_target,
        step,single_pred = False)
    
    X_test, y_test = create_dataset(
        test_X, test_X[['T (°C)']], past_history,future_target, step, 
        single_pred = False)
    
    X_val, y_val = create_dataset(
        val_X, val_X[['T (°C)']], past_history,future_target,step,
        single_pred = False)


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            current_value, input_shape=(X_train.shape[1], X_train.shape[2])))) 

    model.add(tf.keras.layers.RepeatVector(X_train.shape[2]))  
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, activation='softmax')))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(future_target))
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mae')

    # TRAIN
    history = model.fit(X_train, y_train, epochs=15,
                                            batch_size=50,
                                            validation_data=(X_val, y_val))
    
    plt.plot(current_value, history.history['loss'][-1], 'ob')
    plt.plot(current_value, history.history['val_loss'][-1], 'or')
    
    plot_train_history(history, 
                       'Training and Validation Loss (Air Temperature)' +
                       ' - target: ' + str(current_value) + 'h')
    
plt.legend(['Training loss', 'Validation loss'], loc='upper right', ncol=2)
plt.xlabel('Prediction steps')
plt.ylabel('Loss')

##############################################################################
past_history = 120
future_target = 12
step = 1

X_train, y_train = create_dataset(
    train_X, train_X[['T (°C)']], past_history, future_target, 
    step,single_pred = False)

X_test, y_test = create_dataset(
    test_X, test_X[['T (°C)']], past_history,future_target, step, 
    single_pred = False)

X_val, y_val = create_dataset(
    val_X, val_X[['T (°C)']], past_history,future_target, step, 
    single_pred = False)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))) 

model.add(tf.keras.layers.RepeatVector(X_train.shape[2]))  
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='softmax')))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(future_target*y_train.shape[2]))
model.add(tf.keras.layers.Reshape([future_target, y_train.shape[2]]))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mae')

# Train model
history = model.fit(X_train, y_train, epochs=15,
                                            batch_size=50,
                                            validation_data=(X_val, y_val))


plot_train_history(history, 'Training and Validation Loss (Air Temperature)') 

#%% Plot prediction for test set 

for x, y in list(zip(X_test, y_test))[0:3]:
    multi_step_plot(x, y, model.predict(np.expand_dims(x, axis=0))[0],3,
                    train_std, train_mean, 
                    'Prediction for Air Temperature T (°C), 12h window')

#%% Prediction for four Parameters air pressure p (hPa), sun duration 
#   SunDur (h), rainfall Rain (mm) and rletaive humidity Hr (%)


X_train, y_train = create_dataset(
    train_X, train_X[['p (hPa)', 'Rain (mm)', 'Hr (%)', 'SunDur (h)']],
    past_history,future_target,step,single_pred = False)

X_test, y_test = create_dataset(
    test_X, test_X[['p (hPa)', 'Rain (mm)', 'Hr (%)', 'SunDur (h)']],
    past_history,future_target,step,single_pred = False)

X_val, y_val = create_dataset(
    val_X, val_X[['p (hPa)', 'Rain (mm)', 'Hr (%)', 'SunDur (h)']], 
    past_history,future_target,step,single_pred = False)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]))))

model.add(tf.keras.layers.RepeatVector(X_train.shape[2])) 
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(32, activation='softmax')))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(future_target*y_train.shape[2]))
model.add(tf.keras.layers.Reshape([future_target, y_train.shape[2]]))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

# Train model
history = model.fit(X_train, y_train, epochs=15,
                                            batch_size=50,
                                            validation_data=(X_val, y_val))


plot_train_history(history, 'Training and Validation Loss') 

#%% Plot prediction for test set
subtitles = ['Air pressure p (hPa)', 'Rainfall Rain (mm)', 
             'Relative Humidity Hr (%)', 'Sun Duration SunDur (h)']

for x, y in list(zip(X_test, y_test))[100:103]:
    multi_pred_plot(x, y, model.predict(np.expand_dims(x, axis=0))[0],[2,5,6,7],
                    train_std, train_mean, subtitles)

#%% Prediction for two Parameters soil temperature ST (°C) and air temperature 
#   T (°C)

X_train, y_train = create_dataset(
    train_X, train_X[['ST (°C)', 'T (°C)']],
    past_history,future_target,step,single_pred = False)

X_test, y_test = create_dataset(
    test_X, test_X[['ST (°C)', 'T (°C)']],
    past_history,future_target,step,single_pred = False)

X_val, y_val = create_dataset(
    val_X, val_X[['ST (°C)', 'T (°C)']], 
    past_history,future_target,step,single_pred = False)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]))))

model.add(tf.keras.layers.RepeatVector(X_train.shape[2])) 
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(32, activation='softmax')))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(future_target*y_train.shape[2]))
model.add(tf.keras.layers.Reshape([future_target, y_train.shape[2]]))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

# Train model
history = model.fit(X_train, y_train, epochs=15,
                                            batch_size=50,
                                            validation_data=(X_val, y_val))


plot_train_history(history, 'Training and Validation Loss') 

#%% Plot Prediction for test set 

subtitles = ['Soil Temperature ST (°C)', 'Air Temperature T (°C)']

for x, y in list(zip(X_test, y_test))[0:3]:
    multi_pred_plot(x, y, model.predict(np.expand_dims(x, axis=0))[0],[0,3],
                    train_std, train_mean, subtitles)






