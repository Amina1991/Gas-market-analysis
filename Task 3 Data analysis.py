#!/usr/bin/env python
# coding: utf-8

# # 2021 Refinitiv Natural Gas - Analyst Test
# *Prepared by Amina Talipova.*
# ## Below is my approach to solve the basic set of three problems: 
# 3.	Data Analysis.  

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt
import os

# First lets download out datasets and watch their characteristics
file_path = r"C:\Users\User\Desktop\refinitiv\GasPowerDemandNorthCarolina.csv"
df_d = pd.read_csv(file_path, parse_dates=[0],header = 0, dayfirst=False, usecols=range(2))
df_d = pd.DataFrame(df_d)

file_path_1 = r"C:\Users\User\Desktop\refinitiv\temperatures.csv"
df_t = pd.read_csv(file_path_1, parse_dates=[0],header = 0, dayfirst=False, usecols=range(2))
df_t = pd.DataFrame(df_t)

print(df_d.head())
print(df_d.info)

df_t.head()
df_t.info

file_path_2 = r"C:\Users\User\Desktop\refinitiv\HenryHubPrompt.csv"
df_p = pd.read_csv(file_path_2)
df_p.head()

# Then I deal with dates formates to make them proper according to my libraries
df_d.set_index('DATES')
df_t.set_index('DATES')

df_d['DATES'] = pd.to_datetime(df_d['DATES'])
df_d.head (3)

df_t['DATES'] = pd.to_datetime(df_t['DATES'])
df_t.head (3)

# And start modeling with Facebook Prophet package
from fbprophet import Prophet
import logging
logging.getLogger().setLevel(logging.ERROR)

# I also add some specific plot library
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go
import plotly.express as px

# Initialize plotly
init_notebook_mode(connected=True)

# Build the plots to conduct visual statistical analysis
df_d.plot(kind='line',x='DATES',y='G2P [MMcf]', \
          color='darkblue', figsize=(15, 10))
plt.xlabel('')
plt.ylabel('North Carolina gas demand, MMcf', size=12)
plt.show()

# High-frequency data can be rather difficult to analyze. Even with the ability to zoom, 
# it is hard to infer anything meaningful from this chart apart from the prominent seasonality.
# To reduce the noise, resample the post counts down to monthly bins. 

w_df_d = df_d.set_index('DATES').resample('1W').pad()
w_df_d.head(5)

"""
w_df_d.plot(kind='line',y='G2P [MMcf]',color='darkblue', figsize=(15, 10))
plt.xlabel('')
plt.ylabel('North Carolina gas demand, MMcf', size=12)
plt.show()
"""
# Importing Prophet from fbprophet
from fbprophet import Prophet
import logging
logging.getLogger().setLevel(logging.ERROR)

# Here I create new dataet to manipulate with in Prophet 
dfd = df_d
dfd.columns = ['ds', 'y']
dfd.head(5)

# I define the prediction size and create new training dataset
prediction_size = 92
train_dfd = dfd #-prediction_size]
train_dfd.tail(5)

# Here I tell Prophet to make the prediction with defined periods and print 
# the results
model = Prophet(daily_seasonality=False, seasonality_mode='multiplicative')
print("... start fitting")
model.fit(train_dfd)
print("fit is done")
future = model.make_future_dataframe(periods=prediction_size)
new_dfd = model.predict(future)
print(new_dfd.info())
print(new_dfd.tail(10))
print(new_dfd.columns)
fig1 = model.plot(new_dfd)
axes = fig1.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina gas demand forecast, MMsf')
fig2 = model.plot_components(new_dfd)


# After I received the results and see the model is fit, 
# I add temperature as a new regressor

dft = df_t
dft.columns = ['ds', 'y']
print(dft.tail(5))

prediction_size_2 = 0
train_dft = dft #-prediction_size]
train_dft.tail(5)

model_temp = Prophet(daily_seasonality=False, weekly_seasonality=False)
print("... start fitting")
model_temp.fit(train_dft)
print("fit is done")
future_temp = model_temp.make_future_dataframe(periods=prediction_size_2)
new_dft = model_temp.predict(future_temp)
print(new_dft.info())
print(new_dft.tail(10))
print(new_dft.columns)
fig3 = model_temp.plot(new_dft)
axes = fig3.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina Temperature, F')
fig4 = model_temp.plot_components(new_dft);

#train_dfd = train_dft['y']-new_dft['yhat']

#print(train_dfd.tail(30)) 


# Below I merge two my datasest with demand and temperature and drop those 
# dates where I dont have the demand data
#df_t= df_d.stack(dropna=True).reset_index() 
#df_t.tail(5)

# Here I create new dataset to model with temperature surprise effects
df_dt = df_d
df_dt['Temp_surprise'] = train_dft['y']-new_dft['yhat']
df_dt.columns = ['ds', 'y', 'Temp_surprise']
print(df_dt.tail(10))

# I define the prediction size and create new training dataset
prediction_size = 92

train_df_dt = df_dt
print('train_df_dt.tail(5):')
print(train_df_dt.tail(5))

# Here I tell Prophet to make the prediction with defined periods and print the results
model_T_incl = Prophet(daily_seasonality=False)
model_T_incl.add_regressor('Temp_surprise')
print("... start fitting with Temperature surprise")
model_T_incl.fit(train_df_dt)
print("fit is done")
future_wTemp = model_T_incl.make_future_dataframe(periods=prediction_size)
future_wTemp['Temp_surprise'] = train_dft['y']-new_dft['yhat']
print(future_wTemp.tail(30))

new_df_dt = model_T_incl.predict(future_wTemp)
print(new_df_dt.info())
print(new_df_dt.tail(10))
print(new_df_dt.columns)
fig5 = model_T_incl.plot(new_df_dt)
axes = fig5.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina gas demand forecast with Temp, MMsf')
fig6 = model_T_incl.plot_components(new_df_dt)

# The next step is to add regressor from the bonus question. i do it one by one
# and first I add the Henry Hub prices
# Here I create new dataset to model with temperature surprise effects

df_dt = df_d
df_dt['Temp_surprise'] = train_dft['y']-new_dft['yhat']
df_dt.columns = ['ds', 'y', 'Temp_surprise']
print(df_dt.tail(10))