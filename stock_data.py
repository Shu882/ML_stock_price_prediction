import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt


class StockData():
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = yf.download(ticker, start=start, end=end)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data['Scaled Adj Close'] = scaler.fit_transform(self.data['Adj Close'].values.reshape(-1, 1))

    def plot_scaled_close(self):
        self.data['Scaled Adj Close'].plot()
        plt.xlabel('Date')
        plt.ylabel('Scaled Adj Close Price')
        plt.title(f"{self.ticker}: {self.start} to {self.end}  scaled close price")
        plt.show()

    def train_valid_split(self, lookback=20):
        data = []
        data_raw = self.data['Scaled Adj Close'].to_numpy()
        n_total = len(data_raw)
        for index in range(n_total - lookback):
            data.append(data_raw[index:index + lookback + 1])
        data = np.array(data)
        data = data[:, :, np.newaxis]
        n_data = len(data)
        n_test = np.round(n_data * 0.2).astype(int)
        n_train = n_data - n_test
        train = data[:n_train]
        test = data[n_train:]
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = test[:, :-1]
        y_test = test[:, -1]

        # convert all the variables to pytorch tensors
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)

        X_train = X_train.to(torch.float32)
        y_train = y_train.to(torch.float32)
        X_test = X_test.to(torch.float32)
        y_test = y_test.to(torch.float32)

        return (X_train, y_train, X_test, y_test)

    def get_train_test_data_frame(self, lookback=20):
        X_train, y_train, X_test, y_test = self.train_valid_split(lookback=lookback)
        X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
        y_train_see = pd.DataFrame(y_train)
        train_df = pd.concat([X_train_see, y_train_see], axis=1)

        X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0], X_test.shape[1])))
        y_test_see = pd.DataFrame(y_test)
        test_df = pd.concat([X_test_see, y_test_see], axis=1)
        return train_df, test_df

# test code
# ticker = 'NVDA'
# start='2003-01-01'
# end='2022-12-31'
#
# NVDA = StockData(ticker, start, end)
# NVDA.plot_scaled_close()
# X_train, y_train, X_test, y_test = NVDA.train_valid_split()
# print(X_train.size())
# torch.Size([4012, 20, 1])

