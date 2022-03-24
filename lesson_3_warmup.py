#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from typing import List, Dict
import seaborn as sns


# In[57]:


def read_file(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

df = read_file('houses_data.csv')
df = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
df.head()


# In[61]:


df.dtypes


# In[64]:


df.duplicated().sum()


# In[68]:


sns.pairplot(df, hue="Price")


# In[ ]:





# In[97]:


def count_nulls(df) -> pd.DataFrame:
    return df.isnull().sum(axis = 0)

def df_count_nulls(df) -> pd.DataFrame:
    df = pd.DataFrame({'nulls':count_nulls(df), 'value':df.count()})
    df['procent'] = df.nulls / (df.nulls + df.value) * 100
    return df.loc[df['nulls'] > 0]

df_nulls = df_count_nulls(df)

ax = df_nulls.plot.bar(y='procent', label='% of nulls', rot=0)

df_nulls


# In[115]:


rooms = df['Rooms'].unique()
df = df[['Rooms', 'Price']]
#df.plot.bar(y='Price', x='Rooms')

#df_plot = df.plot.bar(y='Price', label='% of nulls', rot=0)

#for i in df_rooms:
#    print(i)
import matplotlib.pyplot as plt


# In[6]:


from sklearn.impute import SimpleImputer

methods = {
    'Car': 'mean',    
}

def simple_imputer(methods):
    for column, method in methods.items():
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy=method)
        df[[column]] = mean_imputer.fit_transform(df[[column]])

df.head()


# In[17]:


from sklearn.impute import KNNImputer

impt = KNNImputer()
impt.fit(df[['BuildingArea', 'Price']])
res = impt.transform(df[['BuildingArea', 'Price']])

df[['BuildingArea', 'Price']] = res
df


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

X = df[['Car', 'BuildingArea']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
result = mean_absolute_error(y_test, y_pred)
result


# In[ ]:




