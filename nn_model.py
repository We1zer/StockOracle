import math
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


class ARIMAModel:


    def __init__(self, dataframe, quote):
        self.quote = quote
        self._df = dataframe
        self._train = None
        self._test = None
        self._predictions = None


    @staticmethod
    def _parser(date):
        return datetime.strptime(date, '%Y-%m-%d')

    def _arima_model(self):
        history = [x for x in self._train]
        predictions = list()

        for t in range(len(self._test)):
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = np.atleast_1d(output)
            predictions.append(yhat[0])
            obs = self._test[t]
            history.append(obs)
        return predictions


    @staticmethod
    def _build_plot_trends(quantity_date):
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(quantity_date)
        plt.savefig('static/Trends.png')
        plt.close(fig)


    def _build_plot_arima(self):
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(self._test, label='Actual Price')
        plt.plot(self._predictions, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/ARIMA.png')
        plt.close(fig)


    def arima_algorithm(self):
        unique_values = self._df["Code"].unique()
        self._df = self._df.set_index("Code")
        for company in unique_values[:10]:
            data = (self._df.loc[company, :]).reset_index()
            data['Price'] = data['Close']
            quantity_date = data[['Price', 'Date']]
            quantity_date.index = quantity_date['Date'].map(lambda x: self._parser(x))
            quantity_date['Price'] = quantity_date['Price'].map(lambda x: float(x))
            quantity_date = quantity_date.fillna(quantity_date.bfill())
            quantity_date = quantity_date.drop(['Date'], axis=1)
            self._build_plot_trends(quantity_date)
            quantity = quantity_date.values
            size = int(len(quantity) * 0.80)
            self._train, self._test = quantity[0:size], quantity[size:len(quantity)]
            self._train = self._train.flatten()
            self._test = self._test.flatten()
            self._predictions = self._arima_model()
            self._build_plot_arima()

            # Метрики
            mae = mean_absolute_error(self._test, self._predictions)
            mse = mean_squared_error(self._test, self._predictions)
            rmse = math.sqrt(mse)
            y_true = np.array(self._test)
            y_pred = np.array(self._predictions)
            non_zero = y_true != 0
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
            r2 = r2_score(self._test, self._predictions)

            print(f"\nTomorrow's {self.quote} Closing Price Prediction by ARIMA: {self._predictions[-2]:.2f}")
            print(f"ARIMA MAE: {mae:.4f}")
            print(f"ARIMA MSE: {mse:.4f}")
            print(f"ARIMA RMSE: {rmse:.4f}")
            print(f"ARIMA MAPE: {mape:.2f}%")
            print(f"ARIMA R²: {r2:.4f}")

            return self._predictions[-2], {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape": mape,
                "r2": r2
            }


class LSTMModel:


    def __init__(self, dataframe, quote):
        self.quote = quote
        self._df = dataframe
        self._x_test = None
        self._x_forecast = None
        self._dataset_train = None
        self._dataset_test = None
        self._real_stock_price = None
        self._predicted_stock_price = None
        self._forecasted_stock_price = None
        self._sc = MinMaxScaler(feature_range=(0, 1))


    def _build_plot_lstm(self):
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(self._real_stock_price, label='Actual Price')
        plt.plot(self._predicted_stock_price, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)

    def _prepare_data(self):
        self._dataset_train = self._df.iloc[0:int(0.8 * len(self._df)), :]
        self._dataset_test = self._df.iloc[int(0.8 * len(self._df)):, :]

        training_set = self._df.iloc[:, 4:5].values
        self._real_stock_price = self._dataset_test.iloc[:, 4:5].values

        self._x_train_list = list()
        self._y_train_list = list()

        training_set_scaled = self._sc.fit_transform(training_set)

        for i in range(7, len(training_set_scaled)):
            self._x_train_list.append(training_set_scaled[i - 7:i, 0])
            self._y_train_list.append(training_set_scaled[i, 0])

        self._x_train = np.array(self._x_train_list)
        self._y_train = np.array(self._y_train_list)

        self._x_train = np.reshape(self._x_train, (self._x_train.shape[0], self._x_train.shape[1], 1))

        self._x_forecast = np.array(self._x_train[-1, 1:])
        self._x_forecast = np.append(self._x_forecast, self._y_train[-1])
        self._x_forecast = np.reshape(self._x_forecast, (1, self._x_forecast.shape[0], 1))

        test_set_scaled = self._sc.transform(self._dataset_test.iloc[:, 4:5].values)

        self._x_test_list = list()
        for i in range(7, len(test_set_scaled)):
            self._x_test_list.append(test_set_scaled[i - 7:i, 0])

        self._x_test = np.array(self._x_test_list)
        self._x_test = np.reshape(self._x_test, (self._x_test.shape[0], self._x_test.shape[1], 1))


    def _lstm_model(self):


        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self._x_train.shape[1], 1)))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        regressor.fit(self._x_train, self._y_train, epochs=25, batch_size=32)
        self._predicted_stock_price = regressor.predict(self._x_test)
        self._predicted_stock_price = self._sc.inverse_transform(self._predicted_stock_price)

        self._forecasted_stock_price = regressor.predict(self._x_forecast)
        self._forecasted_stock_price = self._sc.inverse_transform(self._forecasted_stock_price)

        return regressor

    def lstm_algorithm(self):
        self._prepare_data()
        model = self._lstm_model()
        self._build_plot_lstm()

        self._predicted_stock_price = model.predict(self._x_test)
        self._predicted_stock_price = self._sc.inverse_transform(self._predicted_stock_price)

        if len(self._predicted_stock_price) != len(self._real_stock_price):
            min_len = min(len(self._predicted_stock_price), len(self._real_stock_price))
            self._predicted_stock_price = self._predicted_stock_price[:min_len]
            self._real_stock_price = self._real_stock_price[:min_len]

        y_true = np.array(self._real_stock_price).flatten()
        y_pred = np.array(self._predicted_stock_price).flatten()

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        non_zero = y_true != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = float('nan')
        r2 = r2_score(y_true, y_pred)

        last_sequence = self._x_test[-1]
        future_preds = []

        for _ in range(5):
            next_pred = model.predict(last_sequence.reshape(1, *last_sequence.shape))
            future_price = self._sc.inverse_transform(next_pred)[0][0]
            future_preds.append(future_price)

            next_scaled = self._sc.transform([[future_price]])
            last_sequence = np.append(last_sequence[1:], next_scaled, axis=0)

        lstm_pred = future_preds[0]

        print("LSTM Forecast for Next 5 Days:", future_preds)
        print("Tomorrow's", self.quote, "Closing Price Prediction by LSTM:", lstm_pred)
        print("LSTM RMSE:", rmse)
        print("LSTM MAE:", mae)
        print("LSTM MAPE:", mape)
        print("LSTM R²:", r2)

        return lstm_pred, {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }, future_preds


class LinearRegressionModel:

    def __init__(self, dataframe, quote):
        self.quote = quote
        self._df = dataframe
        self._forecast_days = 5
        self._scale_factor = 1.04
        self._scaler = StandardScaler()
        self._model = LinearRegression(n_jobs=-1)
        self._train_X = None
        self._test_X = None
        self._train_y = None
        self._test_y = None
        self._X_forecast = None
        self._forecast = None
        self._predictions = None

    def _prepare_data(self):
        # Add the 'Close after n days' column
        self._df['Close after n days'] = self._df['Close'].shift(-self._forecast_days)
        df_filtered = self._df[['Close', 'Close after n days']].dropna()

        # Prepare X and y arrays for training
        X = np.array(df_filtered.iloc[:, 0:-1])
        y = np.array(df_filtered.iloc[:, -1]).reshape(-1, 1)

        split_index = int(0.8 * len(X))
        self._train_X = self._scaler.fit_transform(X[:split_index])
        self._test_X = self._scaler.transform(X[split_index:])
        self._train_y = y[:split_index]
        self._test_y = y[split_index:]

        # Ensure only the 'Close' column is used for forecasting (numeric values)
        self._X_forecast = self._scaler.transform(
            np.array(self._df['Close'].iloc[-self._forecast_days:]).reshape(-1, 1))

    def _train_model(self):
        self._model.fit(self._train_X, self._train_y)

    def _build_plot_lr(self):
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(self._test_y, label='Actual Price')
        plt.plot(self._predictions, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/LR.png')
        plt.close(fig)

    def _forecast_prices(self):
        self._forecast = self._model.predict(self._X_forecast) * self._scale_factor
        return self._forecast

    def lin_reg_algorithm(self):
        self._prepare_data()
        self._train_model()

        y_pred = self._model.predict(self._test_X) * self._scale_factor
        self._predictions = y_pred
        self._build_plot_lr()

        lr_pred = self._forecast_prices()[0][0]

        y_true = np.array(self._test_y).flatten()
        y_pred = np.array(self._predictions).flatten()

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Уникаємо ділення на нуль у MAPE
        non_zero = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

        r2 = r2_score(y_true, y_pred)
        forecast_set = self._forecast
        mean = forecast_set.mean()

        print("\n", "Tomorrow's", self.quote, "Closing Price Prediction by Linear Regression:", lr_pred)
        print("Linear Regression RMSE:", rmse)
        print("Linear Regression MAE:", mae)
        print("Linear Regression MAPE:", mape)
        print("Linear Regression R²:", r2)

        return self._df, lr_pred, forecast_set, mean, {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping


class GRUModel:
    def __init__(self, dataframe, quote):
        self.quote = quote
        self._df = dataframe
        self._x_test = None
        self._x_forecast = None
        self._dataset_train = None
        self._dataset_test = None
        self._real_stock_price = None
        self._predicted_stock_price = None
        self._forecasted_stock_price = None
        self._sc = MinMaxScaler(feature_range=(0, 1))

    def _prepare_data(self):
        self._dataset_train = self._df.iloc[0:int(0.8 * len(self._df)), :]
        self._dataset_test = self._df.iloc[int(0.8 * len(self._df)):, :]

        training_set = self._df.iloc[:, 4:5].values
        self._real_stock_price = self._dataset_test.iloc[:, 4:5].values

        self._x_train_list = list()
        self._y_train_list = list()

        training_set_scaled = self._sc.fit_transform(training_set)

        for i in range(7, len(training_set_scaled)):
            self._x_train_list.append(training_set_scaled[i - 7:i, 0])
            self._y_train_list.append(training_set_scaled[i, 0])

        self._x_train = np.array(self._x_train_list)
        self._y_train = np.array(self._y_train_list)

        self._x_train = np.reshape(self._x_train, (self._x_train.shape[0], self._x_train.shape[1], 1))

        self._x_forecast = np.array(self._x_train[-1, 1:])
        self._x_forecast = np.append(self._x_forecast, self._y_train[-1])
        self._x_forecast = np.reshape(self._x_forecast, (1, self._x_forecast.shape[0], 1))

        test_set_scaled = self._sc.transform(self._dataset_test.iloc[:, 4:5].values)

        self._x_test_list = list()
        for i in range(7, len(test_set_scaled)):
            self._x_test_list.append(test_set_scaled[i - 7:i, 0])

        self._x_test = np.array(self._x_test_list)
        self._x_test = np.reshape(self._x_test, (self._x_test.shape[0], self._x_test.shape[1], 1))

    def _gru_model(self):
        regressor = Sequential()
        
        regressor.add(GRU(units=50, return_sequences=True, input_shape=(self._x_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        regressor.add(GRU(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        
        regressor.add(GRU(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        
        regressor.add(GRU(units=50))
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units=1))
        
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        regressor.fit(self._x_train, self._y_train, epochs=50, batch_size=32, callbacks=[early_stopping])
        
        self._predicted_stock_price = regressor.predict(self._x_test)
        self._predicted_stock_price = self._sc.inverse_transform(self._predicted_stock_price)
        
        self._forecasted_stock_price = regressor.predict(self._x_forecast)
        self._forecasted_stock_price = self._sc.inverse_transform(self._forecasted_stock_price)
        
        return regressor

    def _build_plot_gru(self):
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(self._real_stock_price, label='Actual Price')
        plt.plot(self._predicted_stock_price, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/GRU.png')
        plt.close(fig)

    def gru_algorithm(self):
        self._prepare_data()
        model = self._gru_model()
        self._build_plot_gru()

        self._predicted_stock_price = model.predict(self._x_test)
        self._predicted_stock_price = self._sc.inverse_transform(self._predicted_stock_price)

        if len(self._predicted_stock_price) != len(self._real_stock_price):
            min_len = min(len(self._predicted_stock_price), len(self._real_stock_price))
            self._predicted_stock_price = self._predicted_stock_price[:min_len]
            self._real_stock_price = self._real_stock_price[:min_len]

        y_true = np.array(self._real_stock_price).flatten()
        y_pred = np.array(self._predicted_stock_price).flatten()

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        non_zero = y_true != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = float('nan')
        r2 = r2_score(y_true, y_pred)

        last_sequence = self._x_test[-1]
        future_preds = []

        for _ in range(5):
            next_pred = model.predict(last_sequence.reshape(1, *last_sequence.shape))
            future_price = self._sc.inverse_transform(next_pred)[0][0]
            future_preds.append(future_price)

            next_scaled = self._sc.transform([[future_price]])
            last_sequence = np.append(last_sequence[1:], next_scaled, axis=0)

        gru_pred = future_preds[0]

        print("GRU Forecast for Next 5 Days:", future_preds)
        print("Tomorrow's", self.quote, "Closing Price Prediction by GRU:", gru_pred)
        print("GRU RMSE:", rmse)
        print("GRU MAE:", mae)
        print("GRU MAPE:", mape)
        print("GRU R²:", r2)

        return gru_pred, {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }, future_preds




class ImprovedTransformerModel:
    def __init__(self, dataframe, quote):
        self.quote = quote
        self._df = dataframe
        self._x_test = None
        self._x_forecast = None
        self._dataset_train = None
        self._dataset_test = None
        self._real_stock_price = None
        self._predicted_stock_price = None
        self._forecasted_stock_price = None
        self._sc = MinMaxScaler(feature_range=(0, 1))
        self._sequence_length = 5
        self._feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def _add_technical_indicators(self):
        df = self._df.copy()
        
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        df = df.fillna(method='bfill')
        
        self._feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'RSI', 'Momentum']
        self._df = df
        
        return df

    def _prepare_data(self):
        self._add_technical_indicators()
        
        self._dataset_train = self._df.iloc[0:int(0.8 * len(self._df))]
        self._dataset_test = self._df.iloc[int(0.8 * len(self._df)):]

        training_set = self._dataset_train[self._feature_columns].values
        test_set = self._dataset_test[self._feature_columns].values
        training_set_scaled = self._sc.fit_transform(training_set)
        test_set_scaled = self._sc.transform(test_set)

        self._real_stock_price = self._dataset_test[['Close']].values

        self._x_train_list, self._y_train_list = [], []
        for i in range(self._sequence_length, len(training_set_scaled)):
            self._x_train_list.append(training_set_scaled[i - self._sequence_length:i])
            close_idx = self._feature_columns.index('Close')
            self._y_train_list.append(training_set_scaled[i, close_idx])

        self._x_train = np.array(self._x_train_list)
        self._y_train = np.array(self._y_train_list)

        self._x_test_list = []
        for i in range(self._sequence_length, len(test_set_scaled)):
            self._x_test_list.append(test_set_scaled[i - self._sequence_length:i])
        self._x_test = np.array(self._x_test_list)

        self._x_forecast = np.array(training_set_scaled[-self._sequence_length:])
        self._x_forecast = np.reshape(self._x_forecast, (1, self._x_forecast.shape[0], self._x_forecast.shape[1]))

    def _build_transformer_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
        from tensorflow.keras.regularizers import l2
        
        model = Sequential()
        
        inputs = tf.keras.Input(shape=(self._x_train.shape[1], self._x_train.shape[2]))
        
        attention_output = MultiHeadAttention(
            key_dim=32, num_heads=2, dropout=0.1
        )(inputs, inputs)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        ffn_output = Sequential([
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(inputs.shape[-1]),
        ])(attention_output)
        
        x = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(1, kernel_regularizer=l2(0.001))(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model

    def _build_plot_transformer(self):
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(self._real_stock_price, label='Actual Price')
        plt.plot(self._predicted_stock_price, label='Predicted Price')
        plt.legend(loc=4)
        plt.savefig('static/ImprovedTransformer.png')
        plt.close(fig)

    def transformer_algorithm(self):
        self._prepare_data()
        model = self._build_transformer_model()

        early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

        model.fit(
            self._x_train, self._y_train,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        self._predicted_stock_price = model.predict(self._x_test)

        close_idx = self._feature_columns.index('Close')
        dummy = np.zeros((len(self._predicted_stock_price), len(self._feature_columns)))
        dummy[:, close_idx] = self._predicted_stock_price[:, 0]
        self._predicted_stock_price = self._sc.inverse_transform(dummy)[:, close_idx].reshape(-1, 1)

        forecast = model.predict(self._x_forecast)
        dummy_forecast = np.zeros((1, len(self._feature_columns)))
        dummy_forecast[:, close_idx] = forecast[:, 0]
        self._forecasted_stock_price = self._sc.inverse_transform(dummy_forecast)[:, close_idx].reshape(1, 1)
        transformer_pred = self._forecasted_stock_price[0, 0]

        input_seq = self._x_forecast.copy()
        forecast_5d = []

        for _ in range(5):
            next_pred = model.predict(input_seq)[0][0]
            forecast_point = np.zeros((len(self._feature_columns),))
            forecast_point[close_idx] = next_pred

            inv_forecast = self._sc.inverse_transform([forecast_point])[0][close_idx]
            forecast_5d.append(inv_forecast)

            next_input = input_seq[0, 1:, :].copy()
            next_input = np.vstack([next_input, forecast_point])
            input_seq = np.expand_dims(next_input, axis=0)

        self._build_plot_transformer()

        # Метрики
        min_len = min(len(self._predicted_stock_price), len(self._real_stock_price))
        y_true = np.array(self._real_stock_price[:min_len]).flatten()
        y_pred = np.array(self._predicted_stock_price[:min_len]).flatten()

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        non_zero = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else float('nan')
        r2 = r2_score(y_true, y_pred)

        print("Tomorrow's", self.quote, "Closing Price Prediction by Improved Transformer:", transformer_pred)
        print("Improved Transformer RMSE:", rmse)

        return transformer_pred, {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }, forecast_5d
