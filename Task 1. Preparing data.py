#!/usr/bin/python
# -*- coding: utf-8 -*-

## 2021 Refinitiv Natural Gas - Analyst Test
# *Prepared by Amina Talipova.*
### Below is my approach to solve the basic set of three problems:
# 1.....Prepare temperature Data in 6 UK regions

##### I solve this task with two general methods described in the "Rationale and logic implementation" pdf file, where the second method consists of two submethods, each with different methodological approach. Here I start from the most accurate one (Method 3 in pdf Report), and then show two ways of how implement Method 2, described in Report.

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

# open file

path = os.getcwd()
filePath = path + r'\UK Temperatures.csv'


# unify columns

df = pd.read_csv(filePath, parse_dates=[0], header=0, dayfirst=True,
                 usecols=range(7))
df = df.rename(columns={'Unnamed: 0': 'DateTime'})
df = df.set_index('DateTime')

# create dictionary dataframe to store additional data about weather stations coordinates and weight

df_dictionary = pd.DataFrame(np.array([
    [51.758, -1.576, 0.14],
    [50.89, 0.319, 0.1],
    [51.479, -0.449, 0.3],
    [52.9545, -1.1565, 0.13],
    [52.794, -2.663, 0.2],
    [53.166, -0.524, 0.13],
    ]), columns=['Lat', 'Long', 'Weight'])
df_dictionary.index = df.columns

# Method 2.1. is based on the logic that NaN is equal to the average temperature
# of the day before and the day after the missing data at the same time

df1 = pd.read_csv(filePath, index_col=0)
df1.index = pd.to_datetime(df1.index)

# new column with calculated hours to make temporal interpolation

df1['hour'] = df1.index.hour.values

# grouping by hour and interpolating with filling function, where
# bfill take the day before and ffill the day after

df1s = []
for (hour, data) in df1.groupby('hour'):
    df1s.append(((data.fillna(method='bfill')
                + data.fillna(method='ffill'))
                / 2).fillna(method='bfill').fillna(method='ffill'))

# concatinating in new dataframe with interpolated values

df_filled = pd.concat(df1s, axis=0).sort_index()
df_filled['date'] = df_filled.index.date
df_filled.drop(labels=['hour'], axis=1, inplace=True)

# then, group and calculate the mean

df_daily = df_filled.groupby('date').agg(np.mean)

# finally, lets weight the dataframe

df_daily['Weighted_daily_temp'] = df_daily[[
    'Brice Norton',
    'Herstmonceux',
    'Heathrow',
    'Nottingham',
    'Shawbury',
    'Waddington',
    ]].values.dot(df_dictionary.Weight.values[:, np.newaxis])

# looking the output on the plot

df_daily.reset_index().plot(x='date', y='Weighted_daily_temp',
                            color='red', kind='scatter', figsize=(15,
                            10))
plt.show()

# Method 2.2. is based on the logic that the missing NaN is the average
# of nearest observations on the same day
# first, we need to resample the dataframe because the dates and hours of observations deffer

df_new = df.resample('3H').interpolate(method='spline', order=1,
        limit_direction='both')

# and after resampling make grouping by day

df_new = df_new.resample('D').mean()

# check if data frames satisfy to be multiplied to get the weighted average

df_new.values.shape
df_dictionary.Weight.values[:, np.newaxis].shape

# finally, calculate weighted daily average

df_new['daily_temp'] = df_new.values.dot(df_dictionary.Weight.values[:,
        np.newaxis])

# looking the output on the plot

df_new.reset_index().plot(x='DateTime', y='daily_temp', color='red',
                          kind='scatter', figsize=(15, 10))
plt.show()
