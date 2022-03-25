import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


def read_file(path:str, cols:List[str]=None, col_types:Dict[str, str]=None) -> pd.DataFrame:
    if cols is not None and col_types is not None:
        df = pd.read_csv(path, usecols=cols, dtype = col_types)
        return df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        
    else:
        df = pd.read_csv(path, usecols=cols, dtype = col_types)
        return df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    
    
col_name = ['BuildingArea', 'Rooms', 'Price']

col_type = {
    
    'BuildingArea': 'float64',
    'Rooms': 'int16',
    'Price': 'float64'
}

df = read_file('houses_data.csv')

methods = {
    'Car': 'mean',  
    'YearBuilt': 'mean',
    'BuildingArea': 'mean'
}


def simple_imputer(df, methods:Dict[str, str]):
    for column, method in methods.items():
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy=method)
        df[[column]] = mean_imputer.fit_transform(df[[column]])
        return df


df_mean = simple_imputer(df, methods)


df_knn = df

impt = KNNImputer()
df_knn[['BuildingArea', 'YearBuilt', 'Car']] = impt.fit_transform(df[['BuildingArea', 'YearBuilt', 'Car']])


X = df_knn[['Car', 'BuildingArea', 'Distance', 'Bathroom']]
y = df_knn['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

