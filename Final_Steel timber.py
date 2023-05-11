#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as font_manager


# In[2]:


df = pd.read_excel(r"D:\Articles\Steel Timber Article (2)\1.xlsx", sheet_name='Sheet1', header=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, [0, 1, 2, 3]].to_numpy(),
    df.loc[:, 'Ultimate load [kN] '].to_numpy().reshape((-1, 1)),
    test_size=0.3,
    random_state=0
)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[3]:


# Define the K-fold cross-validation parameters
k_folds = 8
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Define the neural network model
model = keras.models.Sequential([
    keras.layers.Dense(32, input_shape=[4,], activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

from keras import backend as K
from sklearn.metrics import r2_score

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Compile the model
model.compile(loss='mse', optimizer='adam' , metrics=[r2_keras])


# In[4]:


r2_scores = []
font = font_manager.FontProperties(family='Times New Roman', size=16)
for i, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
    history=model.fit(X_train_fold, y_train_fold, validation_split = 0.1, epochs=500, batch_size=64, verbose=0)
    
    y_pred_fold = model.predict(X_val_fold)
    ytr_pred_fold = model.predict(X_train_fold)
    r2_fold = round(r2_score(y_val_fold , y_pred_fold),2)
    RMSEte=round(mean_squared_error(y_val_fold , y_pred_fold)**0.5,2)
    r2_foldtr = round(r2_score(y_train_fold , ytr_pred_fold),2)
    RMSEtr=round(mean_squared_error(y_train_fold , ytr_pred_fold)**0.5,2)
    r2_scores.append(r2_fold)
    
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='Train_loss', color='gold')
    ax1.plot(history.history['val_loss'], label='Val_loss', color='green')
    ax1.set_xlabel('Epoch', fontsize=16, fontname='Times New Roman')
    ax1.set_ylabel('Loss', fontsize=16, fontname='Times New Roman')
    ax1.tick_params(axis='y')
    ax1.legend(loc=6)
    
    ax2 = ax1.twinx()
    ax2.plot(history.history['r2_keras'], label='Train_R\u00b2',color='red')
    ax2.plot(history.history['val_r2_keras'], label='Val_R\u00b2',color='blue')
    ax2.set_xlabel('Epoch', fontsize=16, fontname='Times New Roman')
    ax2.set_ylabel('R\u00b2', fontsize=16, fontname='Times New Roman')
    ax2.tick_params(axis='y')
    ax2.legend(loc=7)
    

    # set title and show plot
    plt.title(f'Model performance on k ={i+1}', fontsize=16, fontname='Times New Roman')
    plt.savefig(f"D:/Articles/Steel Timber Article (2)/Figues/Train/{i+1}.svg", format='svg')
    plt.show()


    
    a = min([np.min(y_train_fold), np.min(y_val_fold), 0])
    b = max([np.max(y_train_fold), np.max(y_val_fold), 1])
    plt.scatter(y_train_fold , ytr_pred_fold, s=80, facecolors='red', edgecolors='black',
                label=f'\n Train \n R\u00b2 = {r2_foldtr} \n RMSE = {RMSEtr} \n ')
    plt.legend(fontsize=16, prop={'family': 'Times New Roman'})
    plt.scatter(y_val_fold , y_pred_fold, s=80, facecolors='aqua', edgecolors='black',
                label=f" Valid \n R\u00b2 = {r2_fold}\n RMSE = {RMSEte} ")
    plt.legend(prop={'size': 16, 'family': 'Times New Roman'})

    plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
    plt.legend( prop={'size': 13,'family': 'Times New Roman'},loc=4 )
    plt.title(f'Model results on k = {i+1}' ,fontsize=16 , fontname='Times New Roman')
    plt.xlabel('True Values',fontsize=16, fontname='Times New Roman')
    plt.ylabel('Predictions',fontsize=16, fontname='Times New Roman')
    plt.savefig(f"D:/Articles/Steel Timber Article (2)/Figues/result/{i+1}.svg", format='svg')
    plt.show()

y_pred_test = model.predict(X_test)
r2_test = round(r2_score(y_test, y_pred_test),2)
RMSE=round(mean_squared_error(y_test , y_pred_test)**0.5,2)

y_pred_train = model.predict(X_train)
r2_train = round(r2_score(y_train, y_pred_train),2)
RMSEtr=round(mean_squared_error(y_train, y_pred_train)**0.5,2)

a = min([np.min(y_train_fold), np.min(y_val_fold), 0])
b = max([np.max(y_train_fold), np.max(y_val_fold), 1])
plt.scatter(y_train , y_pred_train, s=80, facecolors='orangered', edgecolors='black',
           label=f'\n Train \n R\u00b2 = {r2_train} \n RMSE = {RMSEtr} \n ')
plt.scatter(y_test , y_pred_test, s=80, facecolors='lime', edgecolors='black',
           label=f'\n Test \n R\u00b2 = {r2_test} \n RMSE = {RMSE} \n ')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.legend( prop={'size': 13,'family': 'Times New Roman'},loc=4  )
plt.title('Model results on test data',fontsize=16,fontname='Times New Roman')
plt.xlabel('True Values',fontsize=16, fontname='Times New Roman')
plt.ylabel('Predictions',fontsize=16, fontname='Times New Roman')
plt.savefig("D:/Articles/Steel Timber Article (2)/Figues/result/9.svg", format='svg')
plt.show()


# In[8]:


df = pd.read_excel(r"D:\Articles\Steel Timber Article (2)\1.xlsx", sheet_name='Sheet1', header=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, [0, 1, 2, 3]].to_numpy(),
    df.loc[:, 'Free end slip [mm]'].to_numpy().reshape((-1, 1)),
    test_size=0.3,
    random_state=0
)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


# Define the K-fold cross-validation parameters
k_folds = 8
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Define the neural network model
model1 = keras.models.Sequential([
    keras.layers.Dense(32, input_shape=[4,], activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

from keras import backend as K
from sklearn.metrics import r2_score

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Compile the model
model1.compile(loss='mse', optimizer='adam' , metrics=[r2_keras])


# In[10]:


r2_scores = []
font = font_manager.FontProperties(family='Times New Roman', size=16)
for i, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
    history=model1.fit(X_train_fold, y_train_fold, validation_split = 0.1, epochs=500, batch_size=64, verbose=0)
    
    y_pred_fold = model1.predict(X_val_fold)
    ytr_pred_fold = model1.predict(X_train_fold)
    r2_fold = round(r2_score(y_val_fold , y_pred_fold),2)
    RMSEte=round(mean_squared_error(y_val_fold , y_pred_fold)**0.5,2)
    r2_foldtr = round(r2_score(y_train_fold , ytr_pred_fold),2)
    RMSEtr=round(mean_squared_error(y_train_fold , ytr_pred_fold)**0.5,2)
    r2_scores.append(r2_fold)
    
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='Train_loss', color='gold')
    ax1.plot(history.history['val_loss'], label='Val_loss', color='green')
    ax1.set_xlabel('Epoch', fontsize=16, fontname='Times New Roman')
    ax1.set_ylabel('Loss', fontsize=16, fontname='Times New Roman')
    ax1.tick_params(axis='y')
    ax1.legend(loc=6)
    
    ax2 = ax1.twinx()
    ax2.plot(history.history['r2_keras'], label='Train_R\u00b2',color='red')
    ax2.plot(history.history['val_r2_keras'], label='Val_R\u00b2',color='blue')
    ax2.set_xlabel('Epoch', fontsize=16, fontname='Times New Roman')
    ax2.set_ylabel('R\u00b2', fontsize=16, fontname='Times New Roman')
    ax2.tick_params(axis='y')
    ax2.legend(loc=7)
    

    # set title and show plot
    plt.title(f'Model performance on k ={i+1}', fontsize=16, fontname='Times New Roman')
    plt.savefig(f"D:/Articles/Steel Timber Article (2)/Figues/Train1/{i+1}.svg", format='svg')
    plt.show()


    
    a = min([np.min(y_train_fold), np.min(y_val_fold), 0])
    b = max([np.max(y_train_fold), np.max(y_val_fold), 1])
    plt.scatter(y_train_fold , ytr_pred_fold, s=80, facecolors='red', edgecolors='black',
                label=f'\n Train \n R\u00b2 = {r2_foldtr} \n RMSE = {RMSEtr} \n ')
    plt.legend(fontsize=16, prop={'family': 'Times New Roman'})
    plt.scatter(y_val_fold , y_pred_fold, s=80, facecolors='aqua', edgecolors='black',
                label=f" Valid \n R\u00b2 = {r2_fold}\n RMSE = {RMSEte} ")
    plt.legend(prop={'size': 16, 'family': 'Times New Roman'})

    plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
    plt.legend( prop={'size': 13,'family': 'Times New Roman'},loc=4 )
    plt.title(f'Model results on k = {i+1}' ,fontsize=16 , fontname='Times New Roman')
    plt.xlabel('True Values',fontsize=16, fontname='Times New Roman')
    plt.ylabel('Predictions',fontsize=16, fontname='Times New Roman')
    plt.savefig(f"D:/Articles/Steel Timber Article (2)/Figues/result1/{i+1}.svg", format='svg')
    plt.show()

y_pred_test = model1.predict(X_test)
r2_test = round(r2_score(y_test, y_pred_test),2)
RMSE=round(mean_squared_error(y_test , y_pred_test)**0.5,2)

y_pred_train = model1.predict(X_train)
r2_train = round(r2_score(y_train, y_pred_train),2)
RMSEtr=round(mean_squared_error(y_train, y_pred_train)**0.5,2)

a = min([np.min(y_train_fold), np.min(y_val_fold), 0])
b = max([np.max(y_train_fold), np.max(y_val_fold), 1])
plt.scatter(y_train , y_pred_train, s=80, facecolors='orangered', edgecolors='black',
           label=f'\n Train \n R\u00b2 = {r2_train} \n RMSE = {RMSEtr} \n ')
plt.scatter(y_test , y_pred_test, s=80, facecolors='lime', edgecolors='black',
           label=f'\n Test \n R\u00b2 = {r2_test} \n RMSE = {RMSE} \n ')
plt.plot([a, b], [a, b], c='gray', lw=1.4, label='y = x')
plt.legend( prop={'size': 13,'family': 'Times New Roman'},loc=4  )
plt.title('Model results on test data',fontsize=16,fontname='Times New Roman')
plt.xlabel('True Values',fontsize=16, fontname='Times New Roman')
plt.ylabel('Predictions',fontsize=16, fontname='Times New Roman')
plt.savefig("D:/Articles/Steel Timber Article (2)/Figues/result1/9.svg", format='svg')
plt.show()


# In[ ]:




