#!/usr/bin/python
# -*- coding: utf-8 -*-

# Refinitiv Natural Gas - Analyst Test
# Prepared by Amina Talipova.*
# Below is my approach to solve Task 3. Data Analysis.

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime as dt
import os

# download datasets

file_path = \
    r"C:\Users\User\Desktop\refinitiv\GasPowerDemandNorthCarolina.csv"
df_d = pd.read_csv(file_path, parse_dates=[0], header=0,
                   dayfirst=False, usecols=range(2))
df_d = pd.DataFrame(df_d)

file_path_1 = r"C:\Users\User\Desktop\refinitiv\temperatures.csv"
df_t = pd.read_csv(file_path_1, parse_dates=[0], header=0,
                   dayfirst=False, usecols=range(2))
df_t = pd.DataFrame(df_t)

file_path_2 = r"C:\Users\User\Desktop\refinitiv\HenryHubPrompt.csv"
df_p = pd.read_csv(file_path_2)

# deal with dates formates

df_d.set_index('DATES')
df_t.set_index('DATES')

df_d['DATES'] = pd.to_datetime(df_d['DATES'])
df_t['DATES'] = pd.to_datetime(df_t['DATES'])

# import Facebook Prophet package

from fbprophet import Prophet
import logging
logging.getLogger().setLevel(logging.ERROR)

# build the plots to make visual analysis

df_d.plot(kind='line', x='DATES', y='G2P [MMcf]', color='darkblue',
          figsize=(15, 10))
plt.xlabel('')
plt.ylabel('North Carolina gas demand, MMcf', size=12)
plt.show()

# create new dataset and rename columns into Prophet standard

dfd = df_d
dfd.columns = ['ds', 'y']

# define the prediction size and create new training dataset
# for the demand autocorrelation model

prediction_size = 92
train_dfd = dfd  # -prediction_size]

# tell Prophet to make the prediction with defined period

model = Prophet(daily_seasonality=False,
                seasonality_mode='multiplicative')
model.fit(train_dfd)
future = model.make_future_dataframe(periods=prediction_size)
new_dfd = model.predict(future)

# show results on the plot

fig1 = model.plot(new_dfd)
axes = fig1.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina gas demand forecast, MMsf')
fig2 = model.plot_components(new_dfd)

# After demand autoregressive model is fit,
# I add temperature as a new regressor and first I do prediction model
# to get the temperature surprise, which will be my regressor
# because demand already containts temperature component and surprise is those
# that in fact affects demand

dft = df_t
dft.columns = ['ds', 'y']
prediction_size_2 = 0
train_dft = dft  # -prediction_size]

model_temp = Prophet(daily_seasonality=False, weekly_seasonality=False,
                     yearly_seasonality=True)  # here I create model and features
model_temp.fit(train_dft)  # fit model with new regressor
future_temp = \
    model_temp.make_future_dataframe(periods=prediction_size_2)  # make prediction
new_dft = model_temp.predict(future_temp)

# show results on the plot

fig3 = model_temp.plot(new_dft)
axes = fig3.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina Temperature, F')
fig4 = model_temp.plot_components(new_dft)

# now create new dataset to model with temperature surprise effects

df_dt = df_d
df_dt['Temp_surprise'] = train_dft['y'] - new_dft['yhat']
df_dt.columns = ['ds', 'y', 'Temp_surprise']

# define the prediction size and create new training dataset

prediction_size = 92
train_df_dt = df_dt

# tell Prophet to make the prediction with defined periods

model_T_incl = Prophet(daily_seasonality=False, yearly_seasonality=True)
model_T_incl.add_regressor('Temp_surprise')
model_T_incl.fit(train_df_dt)
future_wTemp = \
    model_T_incl.make_future_dataframe(periods=prediction_size)
future_wTemp['Temp_surprise'] = train_dft['y'] - new_dft['yhat']

new_df_dt = model_T_incl.predict(future_wTemp)

# show results on the plot

fig5 = model_T_incl.plot(new_df_dt)
axes = fig5.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina gas demand forecast with Temp, MMsf')
fig6 = model_T_incl.plot_components(new_df_dt)

# the next is to include price as regressor and first I do the same
# to calculate the price surprise

dfp = df_p
dfp.columns = ['ds', 'y']

prediction_size_3 = 0
train_dfp = dfp

model_price = Prophet(daily_seasonality=False)
model_price.fit(train_dfp)
future_price = \
    model_price.make_future_dataframe(periods=prediction_size_3)
new_dfp = model_price.predict(future_price)

# show results on the plot

fig7 = model_price.plot(new_dfp)
axes = fig7.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina Price, USD')
fig8 = model_price.plot_components(new_dfp)

# before including the price as a new regressor, I need to combine all
# regressors and depended variable demand into one dataframe and set proper date format

df_d['ds'] = pd.to_datetime(df_d['ds'])
df_d.set_index('ds', inplace=True)
df_dd = df_d.rename(columns={'y': 'demand'})

df_p['ds'] = pd.to_datetime(df_p['ds'])
df_p.set_index('ds', inplace=True)
df_pp = df_p.rename(columns={'y': 'price'})

result = pd.concat([df_dd, df_pp], axis=1, join='outer')
df_result = result.drop(columns=['Temp_surprise'])

# final dataframe with filled NaNs

df_result_filled = df_result.fillna(method='ffill')

# now lets do price autoregression  model to identify price surprise

dfp = df_result_filled.drop(columns=['demand'])
dfpp = dfp.rename(columns={'price': 'y'})

prediction_size_3 = 0
train_dfp = dfpp.reset_index()

model_price = Prophet(daily_seasonality=False, yearly_seasonality=True,
                      weekly_seasonality=True)

model_price.fit(train_dfp)
future_price = \
    model_price.make_future_dataframe(periods=prediction_size_3)
new_dfp = model_price.predict(future_price)

# show results on the plot

fig9 = model_price.plot(new_dfp)
axes = fig9.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('North Carolina price')
fig10 = model_price.plot_components(new_dfp)

# separately identify the price surprice

Price_surprise = pd.DataFrame(train_dfp['y'] - new_dfp['yhat'
                              ]).set_index(train_dfp['ds'])
Price_surprise.columns = ['Price_surprise']
Price_surprise.reset_index()
future_wTemp.set_index('ds', inplace=True)

# now model with price surprise

result_pt = pd.concat([df_result_filled, Price_surprise, future_wTemp],
                      axis=1, join='outer')  # this is dataframe to fit model with
df_pt = result_pt.drop(columns=['price'])

df_pt = df_pt.dropna()
df_pt = df_pt.reset_index() 
df_pt = df_pt.rename(columns={'demand': 'y'})
df_pt = df_pt[['ds', 'y', 'Temp_surprise', 'Price_surprise']]

# create training dataframe where we will define the period to forecast

df_pt_train = df_pt[:-90]

# tell Prophet to make the prediction with defined periods

model_PT_incl = Prophet(daily_seasonality=False,
                        yearly_seasonality=True,
                        weekly_seasonality=True)
model_PT_incl.add_regressor('Temp_surprise')
model_PT_incl.add_regressor('Price_surprise')

model_PT_incl.fit(df_pt_train)

future_wTempPrice = model_PT_incl.make_future_dataframe(periods=90)
future_wTempPrice['Temp_surprise'] = df_pt['Temp_surprise']
future_wTempPrice['Price_surprise'] = df_pt['Price_surprise']

new_df_dpt = model_PT_incl.predict(future_wTempPrice)

# show results on the plot

fig12 = model_PT_incl.plot(new_df_dpt)
axes = fig12.get_axes()
axes[0].set_xlabel('')
axes[0].set_ylabel('Gas Demand, MMcf'
                   )
axes[0].set_title("NC gas demand forecast with Temperature and Price, with COVID-19")

fig13 = model_PT_incl.plot_components(new_df_dpt)
