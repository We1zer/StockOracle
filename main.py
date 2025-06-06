import os
import warnings
import numpy as np

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from flask import Flask, render_template, request

from get_data import get_historical
from nn_model import ARIMAModel, LSTMModel, LinearRegressionModel, GRUModel, ImprovedTransformerModel
from sentiment_analysis import Sentiment

plt.style.use('ggplot')
nltk.download('punkt')
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    quote = request.form['nm']
    try:
        print("Start getting data")
        get_historical(quote)
    except:
        return render_template('index.html', not_found=True)
    else:
        print("Start reading data")
        df = pd.read_csv('' + quote + '.csv')
        today_stock = df.iloc[-1:]
        print("\n", "Today's", quote, "Stock Data: ", today_stock)
        df = df.dropna()
        code_list = []
        for i in range(0, len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2
        
        arima = ARIMAModel(df, quote)
        arima_pred, metrics_arima = arima.arima_algorithm()
        
        lstm = LSTMModel(df, quote)
        lstm_pred, metrics_lstm, lstm_forecast_5d  = lstm.lstm_algorithm()
        
        lin_reg = LinearRegressionModel(df, quote)
        df, lr_pred, forecast_set, mean, metrics_lr = lin_reg.lin_reg_algorithm()
        
        gru = GRUModel(df, quote)
        gru_pred, metrics_gru, gru_forecast_5d = gru.gru_algorithm()
        
        imp_transformer = ImprovedTransformerModel(df, quote)
        imp_transformer_pred, metrics_transformer, transformer_forecast_5d = imp_transformer.transformer_algorithm()
        
        forecasts = [
            lstm_forecast_5d,
            gru_forecast_5d,
        ]

        rmses = [
            metrics_lstm['rmse'],
            metrics_gru['rmse'],
        ]

        forecasts_arr = np.array(forecasts)
        rmses_arr = np.array(rmses)
        eps = 1e-6
        weights = 1 / (rmses_arr + eps)
        weights /= weights.sum()

        weighted_forecast_5d = np.dot(weights, forecasts_arr)
        forecast_vector = weighted_forecast_5d.flatten()
        print("Weighted 5-day forecast based on RMSE:", forecast_vector)
        sentiment = Sentiment(quote, today_stock, mean)
        sentiment.retrieving_tweets_polarity() 
        
        predictions_with_rmse = [
            (arima_pred, metrics_arima['rmse']),
            (lstm_pred, metrics_lstm['rmse']),
            (lr_pred, metrics_lr['rmse']),
            (gru_pred, metrics_gru['rmse']),
            (imp_transformer_pred, metrics_transformer['rmse'])
        ]

        idea, decision, tw_list, tw_pol = sentiment.combined_recommending(predictions_with_rmse)

       

        print("Forecasted Prices for Next 5 days:", forecast_set, sep="\n")
        today_stock = today_stock.round(2)
        
        return render_template('results.html', 
            quote=quote, 
            arima_pred=round(arima_pred, 2),
            lstm_pred=round(lstm_pred, 2),
            lr_pred=round(lr_pred, 2), 
            gru_pred=round(gru_pred, 2),  
            imp_transformer_pred=round(imp_transformer_pred, 2),  
            open_s=today_stock['Open'].to_string(index=False),
            close_s=today_stock['Close'].to_string(index=False),
            tw_list=tw_list, 
            tw_pol=tw_pol, 
            idea=idea, 
            decision=decision,
            high_s=today_stock['High'].to_string(index=False),
            low_s=today_stock['Low'].to_string(index=False),
            vol=today_stock['Volume'].to_string(index=False),
            forecast_set=forecast_vector, 
            error_lr=round(metrics_lr['rmse'], 2), 
            error_lstm=round(metrics_lstm['rmse'], 2),
            error_arima=round(metrics_arima['rmse'], 2),
            error_gru=round(metrics_gru['rmse'], 2),  
            error_imp_transformer=round(metrics_transformer['rmse'], 2),
            metrics_arima=metrics_arima,
            metrics_lstm=metrics_lstm,
            metrics_gru=metrics_gru,
            metrics_lr=metrics_lr,
            metrics_transformer=metrics_transformer
        )

if __name__ == '__main__':
    app.run()