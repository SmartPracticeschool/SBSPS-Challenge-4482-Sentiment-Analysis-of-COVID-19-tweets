import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import plot
import pandas as pd
from plot import Plot
import os
import tweepy
from textblob import TextBlob
import plotly.graph_objects as go
from data_plot_labels import data_plot
from table_plot_labels import table_plot

app = Flask(__name__)
port = int(os.getenv('PORT', 8000))

## Loading the datasets
df = pd.read_csv(r'Model_training/Date_Sentiments.csv')
df1 = pd.read_csv(r'Model_training/Location_Sentiments.csv')
df2 = pd.read_csv(r'Model_training/Location_Date_Sentiments.csv')
df3 = pd.read_csv(r'Data_collection/data/COVID-19_Sentiments.csv')

# For prediction page
output_data = pd.read_csv(r'Model_training/output_data.csv')
train_data = pd.read_csv(r'Model_training/train_data.csv')
valid_data = pd.read_csv(r'Model_training/valid_data.csv')

## Creating list of data points for main plots
data_plot_dict = data_plot(df)
x_list = data_plot_dict['x_list']
y_list = data_plot_dict['y_list']
total_positive = data_plot_dict['total_positive']
total_negative = data_plot_dict['total_negative']
total_neutral = data_plot_dict['total_neutral']
total_positive_num = data_plot_dict['total_num_pos']
total_negative_num = data_plot_dict['total_num_neg']
total_neutral_num = data_plot_dict['total_num_neutral']


## function for accessing total number of each sentiments of every phase
def phase_data(pos_list, neg_list, neu_list, name):
    if name == 'Phase 1':
        pos_sum = sum(pos_list[5:24])
        neg_sum = sum(neg_list[5:24])
        neu_sum = sum(neu_list[5:24])
    elif name == 'Phase 2':
        pos_sum = sum(pos_list[24:43])
        neg_sum = sum(neg_list[24:43])
        neu_sum = sum(neu_list[24:43])
    elif name == 'Phase 3':
        pos_sum = sum(pos_list[43:56])
        neg_sum = sum(neg_list[43:56])
        neu_sum = sum(neu_list[43:56])
    elif name == 'Phase 4':
        pos_sum = sum(pos_list[5:70])
        neg_sum = sum(neg_list[5:70])
        neu_sum = sum(neu_list[5:70])
    
    return [pos_sum, neg_sum, neu_sum]

## getting total number for each phase
ph1_pos, ph1_neg, ph1_neu = phase_data(total_positive_num, total_negative_num,
                                        total_neutral_num, 'Phase 1')
ph2_pos, ph2_neg, ph2_neu = phase_data(total_positive_num, total_negative_num,
                                        total_neutral_num, 'Phase 2')
ph3_pos, ph3_neg, ph3_neu = phase_data(total_positive_num, total_negative_num,
                                        total_neutral_num, 'Phase 3')
ph4_pos, ph4_neg, ph4_neu = phase_data(total_positive_num, total_negative_num,
                                        total_neutral_num, 'Phase 4')
    

## Data for tables
table_dict = table_plot(df1)

sunburst_ploted = plot.sunburst_chart(table_dict['Positive_sentiments'],
                                    table_dict['Negative_sentiments'],
                                    table_dict['Neutral_sentiments']
                        )

# Plots of different phases
phase_1 = plot.phases_plot(x_list, y_list, 'phase 1')
phase_2 = plot.phases_plot(x_list, y_list, 'phase 2')
phase_3 = plot.phases_plot(x_list, y_list, 'phase 3')
phase_4 = plot.phases_plot(x_list, y_list, 'phase 4')



ls = [table_dict, total_positive, total_neutral, total_negative]
ls1 = [phase_1, phase_2, phase_3, phase_4,
        ph1_pos, ph1_neg, ph1_neu,
        ph2_pos, ph2_neg, ph2_neu,
        ph3_pos, ph3_neg, ph3_neu,
        ph4_pos, ph4_neg, ph4_neu
        ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/four', methods=['POST', 'GET'])
def four(): 
    return render_template('sentimeter.html', ls1=ls1, var=[list11,list22,list33, pred_plot], ls=ls)

consumer_key = 'eeSnutOqqknGGGqiso8DPEfdn'
consumer_secret = 'uqKKBzpp96NJTBwEBge9wmsVKEJBdoSuMOmsaQiUphikReuJaH'
access_token = '1276097905363828736-RQfCc3FuSvhdwyYhjpoDfO7QH6Q4gB'
access_token_secret = 'nL8jdSU1j0cuRNiGGum8uTaLrq0blGEwqI8kmfwzAMSsJ'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

list11 = []
list22 = []
list33 = []

new_search = "IndiaFightsCorona+Lockdown -filter:retweets"

tweetsi = tweepy.Cursor(api.search,
                   q=new_search,
                   lang="en",
                   since='2020-07-5', count='20').items(20)

for twee in tweetsi:
    tweet_text = (twee.text)
    list22.append(tweet_text)
    day = str((twee.created_at.day))
    month = str((twee.created_at.month))
    year = str((twee.created_at.year))
    time = str(twee.created_at)
    time = str(time[11:])
    dt = (day + '-' + month + '-' + year + '\n' + time)
    list11.append(dt)
    senti = TextBlob(twee.text)
    senti = (senti.sentiment.polarity)
    if senti <= 0.1:
        list33.append('😡')
    elif senti <= 0.7:
        list33.append('😕')
    else:
        list33.append('😃')
    

pred_plot = plot.predict_plots(x_list[0], output_data, train_data, valid_data)

@app.route('/prediction', methods=['POST', 'GET'])
def prediction(): 
    text = "Enter the tweet!!!"
    if request.method == 'POST':
        message = request.form['message']
        prediction = TextBlob(message)
        prediction = prediction.sentiment.polarity
        if prediction > 0:
            text = f"Sentiment of tweet is positive, {prediction}"
        elif prediction == 0 :
            text = f"Sentiment of tweet is neutral, {prediction}"
        else:
            text = f"Sentiment of tweet is negative, {prediction}"
    return render_template('prediction.html', prediction = text, var=[list11,list22,list33, pred_plot])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)