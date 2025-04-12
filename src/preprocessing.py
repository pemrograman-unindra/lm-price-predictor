import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    df = pd.read_csv(path, parse_dates=["dateTime"])
    df.sort_values("dateTime", inplace=True)
    df = df[["dateTime", "price"]]
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    df["Price_Normalized"] = scaler.fit_transform(df[["price"]])
    return df, scaler

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)