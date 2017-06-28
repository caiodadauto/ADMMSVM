import pandas as pd

def miss_to_mean(data):
    means = data.mean().values
    for column in data.columns:
        data.loc[data[column].isnull(), column] = means[column]

    return data
