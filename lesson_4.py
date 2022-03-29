import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from sklearn.impute import KNNImputer
from scipy.stats import zscore as zscore_outlier, median_abs_deviation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def read_file(path: str, cols: List[str] = None, col_types: Dict[str, str] = None,
              num_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=cols, dtype=col_types)

    if num_only:
        return df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])

    return df


def percentiles_outliers(column: pd.DataFrame) -> bool:
    low, high = np.percentile(column, [10, 90])
    return (column < low) | (column > high)


def iqr_outliers(column: pd.DataFrame) -> bool:
    Q1, Q3 = np.percentile(column, [10, 90])
    IQR = Q3 - Q1
    return (column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR))


def z_score(column: pd.DataFrame) -> bool:
    return zscore_outlier(column) > 3


def modified_z_score_outlier(column: pd.DataFrame) -> bool:
    mad_column = median_abs_deviation(column)
    median = np.median(column)
    mad_score = np.abs(0.6745 * (column - median) / mad_column)
    return mad_score > 3.5


def log_transform(column: pd.DataFrame) -> bool:
    return column.map(lambda x: 0 if x == 0 else np.log10(x)) == None


def score_dataset(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> int:
    regr_model = LinearRegression()
    regr_model.fit(X_train, y_train)
    preds = regr_model.predict(X_test)
    return mean_absolute_error(y_test, preds)


def knn_imputer(df: pd.DataFrame) -> pd.DataFrame:
    knn = KNNImputer()
    knn.fit(df)
    return knn.transform(df)


def outliers_output(df_copy: pd.DataFrame, outliers_methods_dict: Dict[str, Callable]):

    for method_name, method in outliers_methods_dict.items():

        print("\nRunning method: ", method_name)

        df = df_copy.copy()

        print('\nOutliers:\n\n', df.apply(lambda x: method(x)).sum(), '\n\n')

        df = df[df.apply(lambda x: ~method(x))].dropna()

        print(df)

        X = df[['Car', 'Rooms']]
        y = df[['Price']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        print("\nMAE: ", score_dataset(X_train, X_test, y_train, y_test), "\n\n\n\n")


def lesson_4():
    # Select columns to import
    col_name = ['Price', 'Rooms', 'Car']

    # Select types for columns
    col_type = {

        'Price': 'float64',
        'Rooms': 'int16',
        'Car': 'float64'
    }

    df = read_file('houses_data.csv', cols=col_name, col_types=col_type, num_only=False)

    #Count nulls
    df.isnull().sum()

    df.describe()

    #Input values to null
    df[['Car']] = knn_imputer(df[['Car']])

    #Print histogram
    df.hist()

    # Count nulls
    df.isnull().sum()

    # Bar plots
    df[['Rooms']].boxplot()

    df[['Price']].boxplot()

    df[['Car']].boxplot()

    outliers_methods_dict = {
        "log_transform": log_transform,
        "percentiles_outliers": percentiles_outliers,
        "iqr_outliers": iqr_outliers,
        "z_score": z_score,
        "modified_z_score": modified_z_score_outlier
    }

    outliers_output(df, outliers_methods_dict)


if __name__ == '__main__':
    lesson_4()
