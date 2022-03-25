#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import zscore as zscore_outlier, median_abs_deviation


# In[47]:


def read_file(path:str, cols:List[str]=None, col_types:Dict[str, str]=None) -> pd.DataFrame:
    if cols is not None and col_types is not None:
        df = pd.read_csv(path, usecols=cols, dtype = col_types)
        return df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        
    else:
        df = pd.read_csv(path, usecols=cols, dtype = col_types)
        return df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    
    
col_name = ['Price', 'Rooms', 'Car']

col_type = {
    
    'Price': 'float64',
    'Rooms': 'int16',
    'Car': 'float64'
}

df = read_file('houses_data.csv', cols=col_name, col_types=col_type)
df.head()
df.isnull().sum()


# In[48]:


df.describe()


# In[50]:


from sklearn.impute import KNNImputer

impt = KNNImputer()
impt.fit(df[['Car']])
df['Car'] = impt.transform(df[['Car']])
df.hist()
df.isnull().sum()


# In[40]:


df[['Price']].boxplot()


# In[41]:


df[['Rooms']].boxplot()


# In[42]:


df[['Car']].boxplot()


# In[43]:


df_iqr_outliers = df.copy() 

def iqr_outliers(df):
    for x in df:
        q90,q10 = np.percentile(df.loc[:,x],[90,10])
        intr_qr = q90-q10

        max = q90+(1.5*intr_qr)
        min = q10-(1.5*intr_qr)

        df.loc[df[x] < min,x] = np.nan
        df.loc[df[x] > max,x] = np.nan

  
iqr_outliers(df_iqr_outliers)

df_iqr_outliers.isnull().sum()

df_iqr_outliers = df_iqr_outliers.dropna(axis = 0)

df_iqr_outliers.isnull().sum()


# In[44]:


def modified_z_score_outlier(column):
    mad_column = median_abs_deviation(column)
    median = np.median(column)
    mad_score = np.abs(0.6745 * (column - median) / mad_column)
    return mad_score > 3.5


outliers_methods_dict = {
    "z_score": zscore_outlier,
    "mod_z_score": modified_z_score_outlier
}


for method_name, method in outliers_methods_dict.items():
    print("\nRunning method: ", method_name)
    print(df.apply(lambda x: method(x)).sum())


# In[45]:


def df_log(df):
    columns = df.columns

    for col in columns:
        df[col] = np.log(df[col])

    return df

df_log = df_log(df)
df_log


# In[ ]:




