"""
This script process all available fitbit data and consolidate a dataframe for further analysis.
# Input:
# 1. Training data
# 2. date range of training data: today, train_date_range
# 3. User profile (ID, age, exercise choice, ...)
# Output:
# train, test, current week
# 2. dates of missing file (check and re-download)
# setting up basic variables
# run date
Abbreviation:
h: heart rate
s: step
"""
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import glob
import configparser
import os
from datetime import datetime
from xlrd.xldate import xldate_as_tuple
import json
import math
import datetime as dt
from pandas.tseries.offsets import *

###############################
# configuration and constants #
###############################

config = configparser.ConfigParser()
config.read('user.ini')
dir_fitbit = config.get('default', 'user_data_fitbit')
user_id = config.get('default', 'user_id')
# heart_files = glob.glob(dir_fitbit + "heart*.json")
today = config.get('default', 'today')
# for simulation only, for production change to dt.datetime.today().strftime("%Y-%m-%d")
# timestamps in a day
# timestamp_list = pd.date_range(today, periods=96, freq='15min').strftime('%H:%M:%S')
# generate dates for pulling train data ### TODO: find reference for how many weeks of data to pull for train
weeks_to_pull = int(config.get('default', 'weeks_to_pull'))

train_start_date = pd.to_datetime(today) - DateOffset(days=pd.to_datetime(today).dayofweek) - DateOffset(weeks=weeks_to_pull)
train_date_range = pd.date_range(train_start_date, periods=(7 * weeks_to_pull), freq='D').strftime('%Y-%m-%d')

current_start_date = pd.to_datetime(today) - DateOffset(days=pd.to_datetime(today).dayofweek)
current_date_range = pd.date_range(current_start_date, periods=pd.to_datetime(today).dayofweek + 1, freq='D').strftime('%Y-%m-%d')


def prepare_data(date_list, data_dir=dir_fitbit, user_id=user_id):
    # returns fitbit_data['week', 'day_of_week', 'date', 'time', 'time_count', 'heart', 'step']
    list_ = []
    fitbit_data_missing = []
    timestamp_list = pd.date_range(today, periods=96, freq='15min').strftime('%H:%M:%S')
    for d in date_list:
        h_data_exist = True
        s_data_exist = True
        # prepare names of files to load
        # load heart rate json files
        h_filename = data_dir + 'heart' + d + '.json'
        try:
            with open(h_filename) as h:
                h_data = json.load(h)
            h_data = pd.DataFrame.from_dict(h_data['activities-heart-intraday']['dataset'])
            h_data = h_data.set_index('time')
        except ValueError:
            # TODO: add ErrorType
            # file could not be found
            h_data_exist = False
            print('%s could not be found' % h_filename)

        # load step json files
        s_filename = data_dir + 'step' + d + '.json'
        try:
            with open(s_filename) as s:
                s_data = json.load(s)
            s_data = pd.DataFrame.from_dict(s_data['activities-steps-intraday']['dataset'])
            s_data = s_data.set_index('time')
        except ValueError:
            # TODO: add ErrorType
            # file could not be found
            s_data_exist = False
            print('%s could not be found' % s_filename)

        # fill in missing data with 0 TODO: or NaN?
        if h_data_exist & s_data_exist:
            daily_data = pd.DataFrame({'time': timestamp_list,
                                       'time_count': range(len(timestamp_list)),
                                       'heart': 0 * len(timestamp_list),
                                       'step': 0 * len(timestamp_list),
                                       'date': d}, index=timestamp_list)
            for i in daily_data.index:
                try:
                    daily_data.loc[i, 'heart'] = h_data.loc[i, 'value']
                    daily_data.loc[i, 'step'] = s_data.loc[i, 'value']
                except KeyError:
                    next
            list_.append(daily_data)
        else:
            fitbit_data_missing.append(d)
    fitbit_data = pd.concat(list_)

    # get day of the week and week for fitbit_data
    day_of_week = []
    week = []
    for i in fitbit_data.date:
        day_of_week.append(pd.to_datetime(i).dayofweek)
        week.append(pd.to_datetime(i).week)
    fitbit_data['day_of_week'] = day_of_week
    fitbit_data['week'] = week

    # get date_time
    date_time = [fitbit_data['date'][i] + ' ' + fitbit_data['time'][i] for i in range(len(fitbit_data))]
    fitbit_data['date_time'] = date_time
    fitbit_data['user_id'] = user_id
    fitbit_data.index.rename('time_of_day', inplace=True)
    return fitbit_data, fitbit_data_missing


# prepare training data
fitbit_train, fitbit_train_missing = prepare_data(date_list=train_date_range)
fitbit_train.to_csv(config.get('default', 'data_for_analysis') + user_id + '_fitbit_train.csv', encoding='ISO-8859-1')

print(fitbit_train.head())
# prepare data of the current week TODO: separate update of current week's data to a different script
fitbit_current, fitbit_current_missing = prepare_data(date_list=current_date_range)
fitbit_current.to_csv(config.get('default', 'data_for_analysis') + user_id + '_fitbit_current.csv', encoding='ISO-8859-1')

# prepare test data
test_start_date = pd.to_datetime(today) - DateOffset(days=pd.to_datetime(today).dayofweek)
test_date_range = pd.date_range(test_start_date, periods=7, freq='D').strftime('%Y-%m-%d')
fitbit_test, fitbit_test_missing = prepare_data(date_list=test_date_range)
fitbit_test.to_csv(config.get('default', 'data_for_analysis') + user_id + '_fitbit_test.csv', encoding='ISO-8859-1')