"""
1. create demo data
- tend to be sedentary on Monday morning from 10-12 >> suppose to predict and get a notification at 11:30am
"""


import configparser
import copy
import datetime as dt
import math
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
from sklearn.tree import DecisionTreeClassifier

config = configparser.ConfigParser()
config.read('user.ini')
user_id = config.get('default', 'user_id')
fitbit_user_data_dir = config.get('default', 'data_for_analysis')
today = config.get('default', 'today')
day_start_time = float(config.get('default', 'day_start_time')) * 4
day_end_time = float(config.get('default', 'day_end_time')) * 4
# get demo data
fitbit_train_filename = fitbit_user_data_dir + user_id + '_fitbit_train.csv'
fitbit_train = pd.read_csv(fitbit_train_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')

fitbit_train = fitbit_train.reset_index()
window_time_count_list = list(range(40, 49))
for i in fitbit_train.index:
    fitbit_train.loc[(fitbit_train['day_of_week']==0)&(fitbit_train['time_count'].isin(window_time_count_list)), 'step'] = 0

print(fitbit_train)


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


# Method 1: predict sed_prolong
def decision_method_1(train):
    features = ['time_count', 'day_of_week']
    y = train['sed_prolong']
    x = train[features]
    tree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    tree = tree.fit(x, y)
    # check self prediction accuracy
    accuracy = tree.score(x, y)
    # print(tree.feature_importances_)
    # construct dataframe of current week
    monday_current_week = pd.to_datetime(today) - DateOffset(days=pd.to_datetime(today).dayofweek)
    date_range_current_week = pd.date_range(monday_current_week, periods=7,
                                            freq='D').strftime('%Y-%m-%d')
    timestamp_list = pd.date_range(today, periods=96, freq='15min').strftime('%H:%M:%S')
    predicted = pd.DataFrame({'time': list(timestamp_list) * 7,
                              'time_count': list(range(len(timestamp_list))) * 7,
                              'date': np.repeat(date_range_current_week, len(timestamp_list)),
                              'day_of_week': np.repeat(list(range(7)), len(timestamp_list)), })
    # get date_time
    date_time = [predicted['date'][i] + ' ' + predicted['time'][i] for i in range(len(predicted))]
    predicted['date_time'] = date_time
    # predict sedentary bouts
    x_predict = predicted[features]
    predicted_sed_prolong = tree.predict(x_predict)
    predicted['sed_prolong'] = predicted_sed_prolong

    predicted['sed_prolong_start_time'] = 0
    m = 0
    n = 1
    while n < len(predicted):
        if predicted['sed_prolong'][m] == 1:
            if predicted['sed_prolong'][n] == 1:
                # check if consecutive
                t_1 = pd.to_datetime(predicted.loc[n - 1, 'date_time'])
                t_2 = pd.to_datetime(predicted.loc[n, 'date_time'])
                time_delta = t_2 - t_1
                if time_delta == dt.timedelta(minutes=15):
                    predicted.loc[m, 'sed_prolong_start_time'] += 1
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
    print('predicted value of model 1')
    print(accuracy)
    print(predicted[predicted['sed_prolong'] > 0])
    return accuracy, predicted


# Method 2: predict starting time and length of prolonged sedentary bouts
def decision_method_2(train):
    features = ['time_count', 'day_of_week']
    y = train['sed_prolong_start_time']
    x = train[features]
    tree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    tree = tree.fit(x, y)
    # check self prediction accuracy
    accuracy = tree.score(x, y)
    # construct dataframe of current week
    monday_current_week = pd.to_datetime(today) - DateOffset(days=pd.to_datetime(today).dayofweek)
    date_range_current_week = pd.date_range(monday_current_week, periods=7,
                                            freq='D').strftime('%Y-%m-%d')
    timestamp_list = pd.date_range(today, periods=96, freq='15min').strftime('%H:%M:%S')
    predicted = pd.DataFrame({'time': list(timestamp_list) * 7,
                              'time_count': list(range(len(timestamp_list))) * 7,
                              'date': np.repeat(date_range_current_week, len(timestamp_list)),
                              'day_of_week': np.repeat(list(range(7)), len(timestamp_list)), })
    # get date_time
    date_time = [predicted['date'][i] + ' ' + predicted['time'][i] for i in range(len(predicted))]
    predicted['date_time'] = date_time
    # predict sedentary bouts
    x_predict = predicted[features]
    predicted_sed_prolong_start_time = tree.predict(x_predict)
    predicted['sed_prolong_start_time'] = predicted_sed_prolong_start_time
    print('predicted value of model 2')
    print(accuracy)
    print(predicted[predicted['sed_prolong_start_time'] > 0])

    return accuracy, predicted


def decision_method_3(train):
    features = ['time_count', 'day_of_week']
    y = np.where(train['sed_check'] == True, 1, 0)
    x = train[features]
    tree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    tree = tree.fit(x, y)
    # check self prediction accuracy
    accuracy = tree.score(x, y)
    # construct dataframe of current week
    monday_current_week = pd.to_datetime(today) - DateOffset(days=pd.to_datetime(today).dayofweek)
    date_range_current_week = pd.date_range(monday_current_week, periods=7,
                                            freq='D').strftime('%Y-%m-%d')
    timestamp_list = pd.date_range(today, periods=96, freq='15min').strftime('%H:%M:%S')
    predicted = pd.DataFrame({'time': list(timestamp_list) * 7,
                              'time_count': list(range(len(timestamp_list))) * 7,
                              'date': np.repeat(date_range_current_week, len(timestamp_list)),
                              'day_of_week': np.repeat(list(range(7)), len(timestamp_list)), })
    # get date_time
    date_time = [predicted['date'][i] + ' ' + predicted['time'][i] for i in range(len(predicted))]
    predicted['date_time'] = date_time
    # predict sedentary bouts
    x_predict = predicted[features]
    predicted_sed_check = tree.predict(x_predict)
    predicted['sed_check'] = predicted_sed_check  # note that value of sed_check here is [0,1]

    # get sed_prolong_start_time
    # find: 1. starting time of prolonged sedentary behavior, 2. length of each prolonged sedentary time
    predicted = copy.deepcopy(predicted)
    predicted['sed_prolong_start_time'] = 0
    m = 0
    n = 1
    while n < len(predicted):
        if predicted['sed_check'][m] == 1:
            if predicted['sed_check'][n] == 1:
                # check if consecutive
                t_1 = pd.to_datetime(predicted.loc[n - 1, 'date_time'])
                t_2 = pd.to_datetime(predicted.loc[n, 'date_time'])
                time_delta = t_2 - t_1
                if time_delta == dt.timedelta(minutes=15):
                    predicted.loc[m, 'sed_prolong_start_time'] += 1
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
    # filter prolonged sedentary bouts
    predicted = predicted[predicted['sed_prolong_start_time'] >= 5]
    predicted = predicted.reset_index()
    return accuracy, predicted


def get_notification(predicted):
    predicted = copy.deepcopy(predicted)
    predicted = predicted[predicted['sed_prolong_start_time'] >= 5]
    predicted = predicted.reset_index()
    date_time = []
    notification = []
    for i in predicted.index:
        num_msg = math.floor((predicted.loc[i, 'sed_prolong_start_time'] + 1) / 6)
        start_time = pd.to_datetime(predicted.date_time[i]) + DateOffset(hours=1.5)
        for j in range(num_msg):
            date_time.append(start_time)
            notification.append("Would you like to take a break?")
            start_time = start_time + DateOffset(hours=1.5)
    notification = pd.DataFrame({'date_time': date_time, 'notification': notification})
    return notification

sed_train = get_sedentary_behavior(fitbit_train)

prediction_accuracy_1, sed_predicted_1 = decision_method_1(sed_train)
prediction_accuracy_2, sed_predicted_2 = decision_method_2(sed_train)
prediction_accuracy_3, sed_predicted_3 = decision_method_3(sed_train)

prediction_accuracy = [prediction_accuracy_1, prediction_accuracy_2, prediction_accuracy_3]

if prediction_accuracy_1 == max(prediction_accuracy):
    print('model 1 is selected')
    sed_notification = get_notification(sed_predicted_1)
if prediction_accuracy_2 == max(prediction_accuracy):
    print('model 2 is selected')
    sed_notification = get_notification(sed_predicted_2)
if prediction_accuracy_3 == max(prediction_accuracy):
    print('model 3 is selected')
    sed_notification = get_notification(sed_predicted_3)


print(sed_notification)
# sed_train = get_sedentary_behavior(fitbit_train)
