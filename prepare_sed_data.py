# 2016/12
# This script prepares data for sedentary behavior analysis from fitbit JSON file
# output:
# - features: sed_1, sed_2, sed_3, sed_4, sed_5, sed_6, sed_7 (binary representation of SB in 15-min interval)
#             day-1, day-2, day-3, day-4, day-5, day-6, day-7
# - target: sed_prolong

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

config = configparser.ConfigParser()
config.read('user.ini')
user_id = config.get('default', 'user_id')
fitbit_user_data_dir = config.get('default', 'data_for_analysis')
today = config.get('default', 'today')
day_start_time = float(config.get('default', 'day_start_time')) * 4
day_end_time = float(config.get('default', 'day_end_time')) * 4


def get_sedentary_behavior(train):
    # mark out time frame when user is awake yet sedentary
    train['sed_check'] = (train.heart > 0) & (train.step == 0) & (train.time_count >= day_start_time) & (
        train.time_count <= day_end_time)
    try:
        train = train.reset_index()
    except ValueError:
        print("index reset")
    # find: 1. starting time of prolonged sedentary behavior, 2. length of each prolonged sedentary time
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
    # fill in intervals within prolonged sedentary time with 1
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
    # print(train)
    return train


# get training data
fitbit_train_filename = fitbit_user_data_dir + user_id + '_fitbit_train.csv'
fitbit_train = pd.read_csv(fitbit_train_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')
sed_train = get_sedentary_behavior(fitbit_train)

print(sed_train.head())


# get features:
def get_features(n):
    # get features:
    sed_ = []
    for i in sed_train.index:
        # get time series parameters of now
        day_of_week_now = sed_train.day_of_week[i]
        time_count_now = sed_train.time_count[i]
        week_now = sed_train.week[i]
        if (day_of_week_now - n) >= 0:
            # get whether user was sedentary (within a long sedentary bout) on day -1
            try:
                sed_value = sed_train.loc[(sed_train['day_of_week'] == (day_of_week_now - n))
                                            & (sed_train['time_count'] == time_count_now)
                                            & (sed_train['week'] == week_now), 'sed_prolong']
                if len(sed_value) == 0:
                    sed_.append(np.nan)
                else:
                    sed_value = sed_value.reset_index()
                    sed_value = sed_value.loc[0, 'sed_prolong']
                    sed_.append(sed_value)
            except KeyError:
                sed_.append(np.nan)
        else:
            try:
                sed_value = sed_train.loc[(sed_train['day_of_week'] == (7 - abs(day_of_week_now - n)))
                                            & (sed_train['time_count'] == time_count_now)
                                            & (sed_train['week'] == (week_now - 1)), 'sed_prolong']
                if len(sed_value) == 0:
                    sed_.append(np.nan)
                else:
                    sed_value = sed_value.reset_index()
                    sed_value = sed_value.loc[0, 'sed_prolong']
                    sed_.append(sed_value)
            except KeyError:
                sed_.append(np.nan)
    return sed_


for i in range(1,8):
    sed_train = pd.concat([sed_train, pd.DataFrame({i: get_features(i)})], axis=1)

# filter out rows with nan values
sed_train = sed_train.dropna()

# get training data
fitbit_test_filename = fitbit_user_data_dir + user_id + '_fitbit_test.csv'
fitbit_test = pd.read_csv(fitbit_test_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')
sed_test = get_sedentary_behavior(fitbit_test)

# write data to file
sed_train.to_csv(config.get('default', 'data_for_analysis') + user_id + '_sed_train.csv', encoding='ISO-8859-1')
sed_test.to_csv(config.get('default', 'data_for_analysis') + user_id + '_sed_test.csv', encoding='ISO-8859-1')