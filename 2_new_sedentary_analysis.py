from __future__ import print_function
import subprocess
import configparser
import copy
import datetime as dt
import math
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from scipy.stats.stats import spearmanr

# training data: 4 week's data from Jan 14 to Dec 11
# read in fitbit data
# initialize user settings
config = configparser.ConfigParser()
config.read('user.ini')
user_id = config.get('default', 'user_id')
fitbit_user_data_dir = config.get('default', 'data_for_analysis')
today = config.get('default', 'today')
day_start_time = float(config.get('default', 'day_start_time')) * 4
day_end_time = float(config.get('default', 'day_end_time')) * 4

# read in training data
fitbit_train_filename = fitbit_user_data_dir + user_id + '_fitbit_train.csv'
fitbit_train = pd.read_csv(fitbit_train_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')


def get_sedentary_behavior(train):

    # filter out entries when user is awake yet sedentary
    # 1. filter na values 2. filter sleep time
    train['sed_check'] = (train.heart > 0) & (train.step == 0) & (train.time_count >= day_start_time) & (
        train.time_count <= day_end_time)
    # 3. reset index
    try:
        train = train.reset_index()
    except ValueError:
        print("index reset")

    # identify prolonged sedentary behavior
    # 1. find: 1) starting time of prolonged sedentary behavior;
    #          2) length of each prolonged sedentary time
    train['sed_prolong_start_time'] = 0
    m = 0
    n = 1
    while n < len(train):
        if train.loc[m, 'sed_check']:
            if train.loc[n, 'sed_check']:
                # check if consecutive
                t_1 = pd.to_datetime(train.loc[n - 1, 'date'] + ' ' + train.loc[n - 1, 'time'])
                t_2 = pd.to_datetime(train.loc[n, 'date'] + ' ' + train.loc[n, 'time'])
                time_delta = t_2 - t_1
                if time_delta == dt.timedelta(minutes=15):
                    train.loc[m, 'sed_prolong_start_time'] += 1
                    n += 1
                else:
                    m = n
                    n = m + 1
            else:
                m = n + 1
                n = m + 1
        else:
            m = n
            n += 1
    # 2. fill in intervals within prolonged sedentary time with 1
    train['sed_prolong'] = 0
    i = 0
    while i < len(train):
        n_bout = train.sed_prolong_start_time[i]
        if n_bout >= 5:  # >= 90min
            j = i
            while j <= i + n_bout:
                train.loc[j, 'sed_prolong'] = 1
                j += 1
        i += 1
    # contains the following columns:
    # time_of_day date
    # heart step
    # time time_count day_of_week week date_time
    # user_id sed_check sed_prolong_start_time sed_prolong

    # convert date_time to date time object
    train['date_time'] = pd.to_datetime(train['date_time'])
    return train

# get training data
sed_train = get_sedentary_behavior(fitbit_train)

print(sed_train[sed_train['sed_prolong'] == 1])

# transform dataframes for autoregressive modelling
from matplotlib import pyplot
import copy
ar_sed = copy.deepcopy(sed_train)
ar_sed = ar_sed.set_index('date_time')

ar_sed_series = ar_sed.loc[:,'sed_prolong'] # select column to make a series
ar_sed_series = np.asarray(ar_sed_series)
ar_sed_series = ar_sed_series.astype(float)

# plot sedentary behavior
# print(ar_sed_series[:10])
# pyplot.plot(sed_train['sed_prolong'])
# pyplot.plot(ar_sed_series[:(96*7)])
# pyplot.show()
# plot time-lag autocorrelation with 95% CI
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ar_sed_series, lags=450, alpha=.05)
pyplot.show()

# determine features for training decision tree

# transform dataframes to training data structure

# perform decision tree modeling

# predict the next day's sedentary behavior

# display predicted results
from bokeh.plotting import figure, output_file, show


def get_plot(data):
    # output to static HTML file
    output_file("line.html")
    p = figure(plot_width=8000, plot_height=400, x_axis_type='datetime')
    # add a circle renderer with a size, color, and alpha
    p.line(data['date_time'], sed_train['sed_prolong'],line_width=2, color="navy", alpha=0.5)
    show(p)

# generate timing and content for alerts

