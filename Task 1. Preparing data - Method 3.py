#!/usr/bin/python
# -*- coding: utf-8 -*-

## 2021 Refinitiv Natural Gas - Analyst Test
# *Prepared by Amina Talipova.*
### Below is my approach to solve the basic set of three problems:
# 1.....Prepare temperature Data in 6 UK regions

# I solve this task with two general methods described in the "Rationale and logic implementation" pdf file.
# Below is the code for the Method 3, which makes both temporal and spatial relationship.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata as gd
from datetime import datetime as dtdt

# open data file

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

# create new dataframe and add columns with coordinates and weights

df_1 = df.stack(dropna=False).reset_index()
df_1.columns = ['DateTime', 'Station', 'Temperature']

# create new dataframe and merge with temperature, coordinates, and weights
# and apply coordinates and weights to stations. Here we still have NaNs.

df_2 = df_1.set_index('Station').join(df_dictionary)
df_2['Station'] = df_2.index
df_2.reset_index(drop=True, inplace=True)

# now we drop NaNs and save our data without them in new one that I called training

training_df = df_2.dropna().copy()

# I need some parameter to interpolate temporally and I use the number hours

training_df['Hours'] = training_df['DateTime'].sub(dtdt(2016, 1,
        1)).dt.total_seconds().values / 3600

df_2['Hours'] = df_2['DateTime'].sub(dtdt(2016, 1,
        1)).dt.total_seconds().values / 3600

# finally, using griddata and loc functions,
# I interpolate temporally using calculated hours, and spatially using coordinates

empty_value = df_2.Temperature.isna()
interpolated_values = gd(training_df[['Lat', 'Long', 'Hours']].values,
                         training_df['Temperature'].values, df_2[['Lat'
                         , 'Long', 'Hours']][empty_value].values,
                         method='nearest')
df_2.loc[empty_value, 'Temperature'] = interpolated_values

# then I return the original shape of the table,
# aggregate and average data, and delete time for daily observations,

df_2.drop(axis=1, columns=['Lat', 'Long', 'Hours', 'Weight'])
hourly_temp = df_2.set_index(['DateTime', 'Station'
                             ]).Temperature.unstack()
hourly_temp['Date'] = hourly_temp.index.date

# below is the final output: a table with only dates, stations data, and resulted
# weighted average daily temperature

daily_temp = hourly_temp.groupby('Date').agg(np.mean)
daily_temp['weighted'] = \
    daily_temp.values.dot(df_dictionary.Weight.values[:, np.newaxis])
    
# looking the output on the plot
daily_temp.reset_index().plot(x='Date', y='weighted', color='red', kind='scatter', figsize=(15, 10))
plt.show()

