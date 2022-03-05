import numpy as np
import pandas as pd


def export(df, path):
    df_result.to_csv(f"{path}/results", sep=',', encoding='utf-8')

    for col_name in df_result.columns:
        plot = df_result[col_name].plot()
        fig = plot.get_figure()
        fig.savefig(f"{path}/chart_{col_name}.png")


x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

data = [x, y1, y2, y3, x4, y4]

df = pd.DataFrame(data).transpose()

df.columns = ['x', 'y1', 'y2', 'y3', 'x4', 'y4']

df_y = df.iloc[:, lambda df: [1, 2, 3, 5]]

df_result = pd.DataFrame()
df_result['mean'] = df_y.mean()
df_result['std'] = df_y.std()
df_result['var'] = df_y.var()

export(df_result, '../Results')

print(df_result)
