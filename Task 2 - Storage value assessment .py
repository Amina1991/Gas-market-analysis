#!/usr/bin/env python
# coding: utf-8

# # 2021 Refinitiv Natural Gas - Analyst Test
# *Prepared by Amina Talipova.*
# ## Below is my approach to solve the basic set of three problems: 
# 2.	Evaluate natural gas storage; 
#  
# 

# ### I solve this task with the logic described in the "Task 2 explanation.xlsx" supporting document

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from pathlib import Path


# In[2]:


#setting the path to find file, here to demonstrate the code I use Henry Hub prices

path = os.getcwd()
filePath = path + '\HenryHubPrompt.csv'


# In[3]:


# reading file

df = pd.read_csv(filePath)


# In[4]:


# dealing with dates format

df.set_index('Exchange Date',inplace=True)
print(df.head())


# In[5]:


# checking out the max-min for information

print(df.Close.argmax())
print(df.Close.argmin())


# In[6]:


# Lets realize the logic (demonstrated in the Excel file "Task 2 explanation") of the trade strategy when we buy at the 
# record low and sell at the record high prices
# Here the output is an array of positions where even positions are the end of decreasing at the lower peak and 
# odd positions are the end of increasing at the higher peak

def findPeaks(arr):
    '''
    findPeaks () returns the array of indexes fo those days when we sell or buy 
    as list of pairs [[buy_day1, sell_day1],...,[[buy_dayN, sell_dayN]]]
    '''
    indexes = [] 
    lower = True
    prev = arr[0]
    for (pos, val) in enumerate(arr):
        if pos == 0:
            continue
        if (lower and val > prev) or (not lower and val < prev):
            lower = not lower
            indexes.append(pos - 1) # previous day is end of sequence
        else:
            prev = val
  # the last element hasn't been added; it is always the end of an increasing or decreasing sequences
    indexes.append(len(arr) - 1)
    return indexes


# In[7]:


# Finally, we need to write the function that will take our logic of trading and multiply storage capacity and 
# prices to calculate the maximum profit

def mxProf(arr, capacity):
    '''
    mxProf() returns max_profit and the trade strategy as list of pairs [[buy_day1, sell_day1],...,[[buy_dayN, sell_dayN]]]
    inputs: 
    arr - array of prices
    capacity - storage capacity
    '''
    profits = 0
    if len(arr) < 2: # here is writen the condtion that we cannot sell and buy on one day 
        return profits, []
    seqs = findPeaks(arr)

    if len(seqs) == 1: # if our array is dectreasing in a constant way; or minimize loss 
        prev = arr[1]
        ind = 1
        maxDiff = arr[1] - arr[0]
        for (pos, val) in enumerate(arr):
            if pos == 0 or pos == 1:
                continue
            if val - prev > maxDiff:
                maxDiff = val - prev
                ind = pos
            prev = val
        return [[ind - 1, ind]]
  
    if len(seqs) % 2 == 1: # define when the lowering is ended
        seqs = seqs[:len(seqs)-1]
  
  # here we sell at the highest peak, buy at the the lowest, and wait between
    days = []
    for i in range(1, len(seqs), 2):
        days.append([seqs[i-1], seqs[i]])
        profits = profits + capacity*(arr[seqs[i]] - arr [seqs[i-1]])
    return profits, days


# In[8]:


# finaly, check if the function to find trading strategy and maximized profit works

mxProf(df.Close.values, 200)


# In[ ]:




