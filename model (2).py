# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:07:47 2022

@author: user
"""

# importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import requests
import json


# In[2]:


# importing data
df =pd.read_csv('C:/Users/user/Downloads/CarPrice_Assignment.csv')


# ### Wrangling the data

# #### Evaluating for missing data

# In[3]:


missing_data = df.isnull()
missing_data.head(5)


# #### Counting missing values

# In[4]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# #### Evaluating for duplicate values

# In[5]:


bool_series = df.duplicated()
bool_series.head(5)


# #### Removing duplicate rows

# In[6]:


# Use the keep parameter to consider only the first instance of a duplicate row to be unique
bool_series = df.duplicated(keep='first')
print('Boolean series:')
print(bool_series)
print('\n')
print('DataFrame after keeping only the first instance of the duplicate rows:')

# The `~` sign is used for negation. It changes the boolean value True to False and False to True.

df[~bool_series]


# In[7]:



df['make'] = df['CarName'].str.split().str[0]
print(df['make'])


# In[8]:


df.info()


# ### Feature Transformation
# We shall turn our categorical values into numerical values(One hot encoding)

# In[9]:


df.info()


# In[10]:


df = pd.get_dummies(df, columns = ['fueltype','aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype','cylindernumber', 'fuelsystem','make'])
print(df)


# ### Training and Testing
# We shall drop the price column and use the rest of the columns as regressors

# In[11]:


x_data=df.drop('price', axis=1)
x_data=df.drop('CarName', axis=1)
y_data = df['price']


# now, we shall randomly split our data into train and test data

# In[12]:



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=0)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[13]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[14]:


y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
df


# In[15]:


from sklearn.linear_model import Ridge
RidgeModel = Ridge(alpha = 0.5)
RidgeModel.fit(x_data, y_data)
yhat = RidgeModel.predict(x_data)


# In[16]:


df = pd.DataFrame({'Real Values':y_data, 'Predicted Values':yhat})
df


# In[17]:


pickle.dump(RidgeModel, open('model.pkl','wb'))

