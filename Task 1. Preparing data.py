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


# In[4]:


# analyze dataframe

df.info()


# In[5]:


# analyze dataframe

df.head(10)


# ### Method 2.1.  

# In[ ]:


df1 = pd.read_csv(filePath, index_col=0)
df1.index = pd.to_datetime(df1.index)
df1.head()


# In[ ]:


df1['hour'] = df1.index.hour.values
df1


# In[ ]:


df1s = []
for hour, data in df1.groupby('hour'):
    df1s.append(
        ((data.fillna(method='bfill') + 
         data.fillna(method='ffill'))/2).
        fillna(method='bfill').
        fillna(method='ffill')
    )
df_filled = pd.concat(df1s, axis=0).sort_index()


# In[ ]:


df_filled['date'] = df_filled.index.date
df_filled.drop(labels=['hour'], axis=1, inplace=True)
df_daily=df_filled.groupby('date').agg(np.mean)
print(df_daily.head(10))


# In[ ]:




df_daily.info()


# In[ ]:


df_daily.values.shape


# In[ ]:


df_daily['Weighted_daily_temp'] = df_daily[['Brice Norton','Herstmonceux', 'Heathrow','Nottingham', 'Shawbury','Waddington']].values.dot(df_dictionary.Weight.values[:, np.newaxis])
print(df_daily.head(20))


# In[ ]:



df_daily.reset_index().plot(x='date', y='Weighted_daily_temp', color='red', kind='scatter', figsize=(15, 10))
plt.show()


# ### Method 2.2.

# In[11]:


# Method 2.2 First, we need to resample the dataframe because the dates and hours of observations defferent 


df_new = df.resample('3H').interpolate(method = 'spline', order = 1, limit_direction='both')

print(df_new.head(20))
print(df_new.info())


# In[12]:


# and after resampling make grouping by day

df_new = df_new.resample('D').mean()
print(df_new.head(20))
print(df_new.info())


# In[13]:


# analyze dataframe

df_new.info()


# In[14]:


# check if data frames satisfy to be multiplied to get the weithed average

df_new.values.shape
df_dictionary.Weight.values[:, np.newaxis].shape


# In[15]:


# finally, calculate weighted daily average

df_new['daily_temp'] = df_new.values.dot(
    df_dictionary.Weight.values[:, np.newaxis]
)


# In[16]:


print(df_new.head(10))


# In[19]:


df_new.reset_index().plot(x='DateTime', y='daily_temp', color='red', kind='scatter', figsize=(15, 10))
plt.show()


# In[ ]:




