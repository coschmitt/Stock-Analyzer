import yfinance as yf
import pandas_datareader as pdr
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from os import path

from tensorflow.python.keras.models import model_from_json

class MakeModel:

    def __init__(self, ticker, path):
        self.normalizer = None
        self.path = path
        self.data = pd.DataFrame()         # raw data from yahoo finance api
        self.model = None                  # model after the train_model function is called
        self.X_train = []                  # X data for training (Open, Close, High, Low and volume)
        self.Y_train = []                  # Y data for training (Next day stock price)
        self.X_test = []                   # X data for testing model
        self.predicted_stock_price = []    # predicted stock price after testing the model
        self.ticker = ticker               # ticker symbol for the stock

    def get_data(self):
        yf.pdr_override()

        start_date = pd.to_datetime('2010-10-01')
        end_date = pd.to_datetime('2020-10-01')

        data = pdr.get_data_yahoo(
            self.ticker,
            start=start_date,
            end=end_date
        )

        self.data = data

        # get first 90% of data for training set
        training_set = data[0:int(len(data) * 0.9)]['Open']
        testing_set = data[int(len(data) * 0.9):]['Open']

        training_set = pd.DataFrame(training_set)
        testing_set = pd.DataFrame(testing_set)


        # initialize  object that will be used to normalize data
        sc = MinMaxScaler(feature_range=(0, 1))

        # normalize data
        scaled_training_set = sc.fit_transform(training_set)

        self.normalizer = sc

        X_train = []
        Y_train = []

        for i in range(60, len(training_set)):
            X_train.append(scaled_training_set[i - 60:i, 0])
            Y_train.append(scaled_training_set[i, 0])

        self.X_train, self.Y_train = np.array(X_train), np.array(Y_train)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        print(training_set)

        # get test
        dataset_total = pd.concat((training_set, testing_set), axis=0)
        inputs = dataset_total[len(dataset_total) - len(testing_set) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, len(testing_set)):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        self.X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    def train_model(self):
        regressor = Sequential()

        # adding LSTM layers
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=50, return_sequences=False))
        regressor.add(Dropout(0.2))

        # adding the output layer
        regressor.add(Dense(units=1))

        #  compiling the the RNN and fitting the model
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressor.fit(self.X_train, self.Y_train, epochs=100, batch_size=32)
        # have model to disk for later use
        self.model = regressor
        model_json = regressor.to_json()
        with open(self.path+"/json/" + self.ticker + "-model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        regressor.save_weights(self.path+"/h5/" + self.ticker + "-model.h5")
        print("Saved model to disk")

    def test_model(self):
        testing_set = pd.DataFrame(self.data[int(len(self.data)*0.9):]['Open'])
        real_stock_price = self.data[int(len(self.data) * 0.9):].iloc[:len(testing_set) - 60, 1:2].values

        if not path.exists(self.path+"/json/"+self.ticker + "-model.json"):
            predicted_stock_price = self.model.predict(self.X_test)

        else:
            json_file = open(self.path+"/json/" + self.ticker + '-model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            regressor = model_from_json(loaded_model_json)

            # load weights into new model
            regressor.load_weights(self.path+"/h5/"+self.ticker+"-model.h5")
            print("Loaded model from disk")

            # evaluate loaded model on test data
            regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
            predicted_stock_price = regressor.predict(self.X_test)


        predicted_stock_price = self.normalizer.inverse_transform(predicted_stock_price)
        self.predicted_stock_price = pd.DataFrame(predicted_stock_price)

        plt.plot(real_stock_price, color='red', label='Real ' +self.ticker+ ' Stock Price')
        plt.plot(self.predicted_stock_price, color = 'blue', label = 'Predicted ' + self.ticker + ' Stock Price')
        plt.title(self.ticker+' Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(self.ticker+' Stock Price')
        plt.legend()
        plt.show()
