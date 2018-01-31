import pandas as pd

def miss_to_mean(data, encode = None):
    means = data.mean().values

    if encode is None:
        for column in data.columns:
            data.loc[data[column].isnull(), column] = means[column]
    else:
        for column in data.columns:
            data.loc[data[column] == encode, column] = means[column]


    return data
