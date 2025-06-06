import time
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
from re import sub
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
from preprocessor import clean


class Sentiment:
    def __init__(self, quote, today_stock, mean):
        self.today_stock = today_stock
        self.quote = quote
        self.mean = mean
        self._user = None
        self._positive = 0
        self._negative = 1
        self._neutral = None
        self._global_polarity = 0
        self._tw_pol = None
        self._tw_list = []

    def _authentication(self):
        auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
        self.user = tweepy.API(auth)

    def _build_plot_tweets(self):
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [self._positive, self._negative, self._neutral]
        explode = (0, 0, 0)
        
        colors = ['green', 'red', 'gray']  
        
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        fig1, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
        
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        
        ax1.axis('equal')  
        plt.tight_layout()
        plt.savefig('static/SA.png')
        plt.close(fig)


    def retrieving_tweets_polarity(self):
        stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
        stock_full_form = stock_ticker_map[stock_ticker_map['Ticker'] == self.quote]
        symbol = stock_full_form['Name'].to_list()[0][0:12]
        self._authentication()

        client = tweepy.Client(bearer_token=ct.bearer_token)  
        query = f"{symbol} lang:en -is:retweet"  

        
        retry_count = 0
        while retry_count < 5:  
            try:
                tweets = client.search_recent_tweets(query=query, tweet_fields=["text"], max_results=ct.num_of_tweets)
                break  
            except tweepy.errors.TooManyRequests as e:
                retry_count += 1
                reset_time = int(e.response.headers['x-rate-limit-reset']) - int(time.time())
                print(f"Rate limit exceeded, retrying in {reset_time} seconds...")
                time.sleep(reset_time + 10) 

        tweet_list = []
        count = 20
        for tweet in tweets.data:  
            tw2 = tweet.text
            tw = tweet.text
            tw = clean(tw)
            tw = sub('&amp;', '&', tw)
            tw = sub(':', '', tw)
            tw = tw.encode('ascii', 'ignore').decode('ascii')
            blob = TextBlob(tw)
            polarity = 0
            for sentence in blob.sentences:
                polarity += sentence.sentiment.polarity
                if polarity > 0:
                    self._positive += 1
                if polarity < 0:
                    self._negative += 1
                    self._global_polarity += sentence.sentiment.polarity
            if count > 0:
                self._tw_list.append(tw2)
            tweet_list.append(Tweet(tw, polarity))
            count -= 1

        if len(tweet_list) != 0:
            self._global_polarity = self._global_polarity / len(tweet_list)
        self._neutral = 20 - self._positive - self._negative
        if self._neutral < 0:
            self._negative += self._neutral
            self._neutral = 20

        print("\n", f"Positive Tweets: {self._positive}", f"Negative Tweets: {self._negative}",
              f"Neutral Tweets: {self._neutral}")
        if self._global_polarity > 0:
            print("Tweets Polarity: Overall Positive")
            self._tw_pol = "Overall Positive"
        else:
            print("Tweets Polarity: Overall Negative")
            self._tw_pol = "Overall Negative"

    
    def combined_recommending(self, predictions_with_metrics):
       
        filtered = [(float(pred), rmse) for pred, rmse in predictions_with_metrics if rmse and rmse < 1e6]
        if not filtered:
            print("No valid predictions available.")
            return "UNCERTAIN", "HOLD", self._tw_list, self._tw_pol

        weights = [1 / rmse for _, rmse in filtered]
        total_weight = sum(weights)
        weighted_prediction = sum(pred * w for (pred, _), w in zip(filtered, weights)) / total_weight

        current_price = float(self.today_stock.iloc[-1]['Close'])

        if weighted_prediction > current_price and self._global_polarity > 0:
            idea = "RISE"
            decision = "BUY"
        elif weighted_prediction < current_price and self._global_polarity <= 0:
            idea = "FALL"
            decision = "SELL"
        else:
            idea = "UNCERTAIN"
            decision = "HOLD"

        print(f"\n[RECOMMENDATION] Weighted prediction: {weighted_prediction:.2f}, Current price: {current_price:.2f}")
        return idea, decision, self._tw_list, self._tw_pol
