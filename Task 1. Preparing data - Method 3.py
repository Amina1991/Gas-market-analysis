#!/usr/bin/env python
# coding: utf-8

# # 2021 Refinitiv Natural Gas - Analyst Test
# *Prepared by Amina Talipova.*
# ## Below is my approach to solve the basic set of three problems: 
# 1.	Prepare temperature Data in 6 UK regions

# #### I solve this task with two general methods described in the "Rationale and logic implementation" pdf file, where the second method consists of two submethods, each with different methodological approach. Here I start from the most accurate one (Method 3 in pdf Report), and then show two ways of how implement Method 2, described in Report. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from scipy.interpolate import griddata as gd
from datetime import datetime as dtdt

# In[2]:


# open data file

path = os.getcwd()
filePath = path + r'\UK Temperatures.csv'
print(filePath)


# In[3]:


# unify columns' titles and set daytime column as index 

df = pd.read_csv(filePath, parse_dates=[0],header = 0, dayfirst=True, usecols=range(7))
df = df.rename(columns = {"Unnamed: 0" : "DateTime"})
df = df.set_index('DateTime')
#df=df[0:500].copy()

# In[4]:


# analyze dataframe

df.info()


# In[5]:


# analyze dataframe

df.head(10)


# #### Method 3. To have both temporal and spatial relationship accountancy, we need to interpolate across both columns and rows and then calculate the weighted average. This dual interpolation requires additional data about weather station coordinates. 

# In[6]:


# create new dataframe, reshaping it from the original one so that we can add columns with coordinates and weights

df_1= df.stack(dropna=False).reset_index()
df_1.columns = ['DateTime', 'Station', 'Temperature']


# In[7]:


# analyze dataframe

df_1.head(10)


# In[8]:


# create dictionary dataframe to store additional data about weather stations coordinates and weight

df_dictionary = pd.DataFrame(np.array([[51.758,-1.576,0.14], [50.89,0.319,0.1], [51.479,-0.449,0.3], [52.9545,-1.1565,0.13],[52.794,-2.663,0.2],[53.166,-0.524,0.13]]),
                   columns=['Lat', 'Long', 'Weight'])
df_dictionary.index= df.columns
print (df_dictionary)


# In[9]:


# create new dataframe where we merge two previous dataframes with temperature, coordinates, and weights 
# and apply coordinates and weights to stations. Here we still have NaNs. 

df_2 = df_1.set_index('Station').join(df_dictionary)
df_2['Station'] = df_2.index
df_2.reset_index(drop=True, inplace=True)
print(df_2.head(30))
print(df_2.columns)


# In[10]:


# now we drop NaNs and save our data without them in new one that I called training

training_df = df_2.dropna().copy()
print(training_df.head(10))
print()

# in other dataframe I called for convenience filling (originally it is just old one with missing data), which we will fill  
# filling_df = df_2
#print(filling_df.columns)


# In[11]:


# to fill it, I need some parameter to rely on and I use the number of seconds as I have this data everywhere, unlike other types of data

training_df['Hours'] = (training_df['DateTime'].sub(dtdt(2016, 1, 1)).dt.total_seconds().values)/3600

df_2['Hours'] = (df_2['DateTime'].sub(dtdt(2016, 1, 1)).dt.total_seconds().values)/3600


# In[12]:


# finally, using griddata and loc functions, finally, I interpolate the dataframe using the data in the training one. 
# Specifically, I interpolate temporally using calculated seconds, and spatially using the coordinates 

# from scipy.interpolate import griddata




print('interpolation started')
#[La, Lo, Ho] = training_df.loc[:,['Lat','Long','Hours']].values
#[mLa, mLo, mHo] = (filling_df.loc[:,['Lat', 'Long', 'Hours']].values)

#Z = training_df.loc[:,"Temperature"].values

df_2['NewTemp']=gd((training_df[['Lat','Long','Hours']].values),
         (training_df["Temperature"].values),
         (df_2[['Lat', 'Long', 'Hours']].values), 
         method='linear')





# to regroup again later, I set two columns as indexes

df_2.set_index(["DateTime","Station"], inplace=True)


# In[ ]:


# analyze dataframe

print(df_2.head(10))


# In[ ]:


# then I return the original type of the table, aggregate and average data, and delete time for daily observations
df_2.drop(axis = 1, columns='Temperature')
print(df_2.head(10))
hourly_temp = df_2.NewTemp.unstack()
hourly_temp["Date"] = hourly_temp.index.date
daily_temp = hourly_temp.groupby("Date").agg(np.mean)


# In[ ]:


daily_temp['weighted'] = daily_temp.values.dot(
    df_dictionary.Weight.values[:, np.newaxis]
)

print('--------------------')
print('The daily temperature via temporal-spatial interpolation:')
print('--------------------')
print(daily_temp['weighted'].head(20))


# ### Method 1.1.  

# In[ ]:


df1 = pd.read_csv(filePath, index_col=0)
df1.index = pd.to_datetime(df1.index)
df1.head()


# ### Method 1.2.

# In[ ]:


# Method 2. First, we need to resample the dataframe and after resampling make grouping by day

df_new = df.resample('3H').interpolate(method = 'spline', order = 1, limit_direction='both')

print(df_new.head(20))
print(df_new.info())


print('--------------')

df_new = df_new.resample('D').mean()
print(df_new.head(20))
print(df_new.info())


# In[ ]:





# In[ ]:


df_new.info()


# In[ ]:


#check if data frames satisfy to be multiplied to get the weithed average
df_new.values.shape
df_dictionary.Weight.values[:, np.newaxis].shape


# In[ ]:


#calculate weighted average
df_new.values.dot(
    df_dictionary.Weight.values[:, np.newaxis]
)


# In[ ]:




