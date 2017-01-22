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

# print(sed_train[sed_train['sed_prolong'] == 1])

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
# plot_acf(ar_sed_series, lags=450, alpha=.05)
# pyplot.show()


# get autocorrelation coefficient
import numpy as np
from scipy.optimize import leastsq
import pylab as plt

import statsmodels.tsa.stattools as smtsa

acf = smtsa.acf(sed_train['sed_prolong'], nlags=len(ar_sed_series), unbiased=True)
# print(acf[[96, 192, 288]])
# for i in range(len(acf)):
#     if acf[i] > .5:
#         print('the %d th %s lag has coefficient %f' % (range(len(acf))[i], str(ar_sed.index[i]), acf[i]))
# determine features for training decision tree
# only use one week's data points
acf = acf[:1000]
t = np.array(range(len(acf)))
data = acf  # ndarray
data_guess = [3 * np.std(data) / (2 ** 0.5), 2 * np.pi / 100, np.pi / 2, np.mean(data), -0.0025, -0.0025]  # std, freq, phase, mean, slope


def get_sin_model(guess, t, data):
    # This functions models
    guess_std = guess[0]
    guess_freq = guess[1]
    guess_phase = guess[2]
    guess_mean = guess[3]
    guess_slope1 = guess[4]
    guess_slope2 = guess[5]
    data_first_guess = guess_std * np.exp(guess_slope1*t) * np.sin(np.exp(guess_slope2*t)* guess_freq * t + guess_phase) + guess_mean
    optimize_func = lambda x: x[0] * np.exp(x[4]*t) * np.sin(np.exp(x[5]*t) * x[1] * t + x[2]) + x[3] - data
    est_std, est_freq, est_phase, est_mean, est_slope1, est_slope2 = leastsq(optimize_func, [guess_std, guess_freq, guess_phase, guess_mean, guess_slope1, guess_slope2])[0]
    # recreate the fitted curve using the optimized parameters
    data_fit = est_std * np.exp(est_slope1*t) * np.sin(np.exp(est_slope2*t)*est_freq * t + est_phase) + est_mean

    print('fitted std, freq, phase, mean, slope are %f %f %f %f %f %f' % (est_std, est_freq, est_phase, est_mean, est_slope1, est_slope2))
    print("thus, the desired interval should be approximately %f day" % (2*np.pi/est_freq/96))
    estimation = [est_std, est_freq, est_phase, est_mean,est_slope1, est_slope2]
    # plot results
    plt.plot(data, '.')
    plt.plot(data_fit, label='after fitting')
    plt.plot(data_first_guess, label='first guess')
    plt.legend()
    plt.show()

    return estimation

get_sin_model(data_guess, t, data)

def days_to_include(auto_corr, interval):
    pointer = interval
    feature_position_list = []
    while pointer <= len(auto_corr):

        if auto_corr[pointer] >= 0.4:  #TODO: how to determine the cut-off value for correlation coefficient
            feature_position_list.append(pointer)

        pointer += interval
    print("this is the features' position list:")
    print(feature_position_list)
    return feature_position_list

days_to_include(acf, 96)
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

