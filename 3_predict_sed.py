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

# sed_test_filename = fitbit_user_data_dir + user_id + '_sed_test.csv'
# sed_test = pd.read_csv(sed_test_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')


# print(sed_test.head())

sed_train_filename = fitbit_user_data_dir + user_id + '_sed_train.csv'
sed_train = pd.read_csv(sed_train_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')
sed_train = sed_train.reset_index()

print(len(sed_train))
sed_test = sed_train[-96*20:-96]
sed_test = sed_test.reset_index()
sed_train = sed_train[:(len(sed_train)-96*20)]


def predict_sed(train, test):
    features = ['1', '2', '3']
    y = train['sed_prolong']
    x = train[features]
    tree = DecisionTreeClassifier(min_samples_split=50)
    tree = tree.fit(x, y)
    # check self prediction accuracy
    accuracy = tree.score(x, y)
    print(accuracy)
    # predict sedentary bouts
    x_predict = test[features]
    y_predicted = tree.predict(x_predict)
    result_compare = test
    result_compare['predicted'] = y_predicted
    # result_compare = pd.DataFrame({'predicted': y_predicted, 'value': test['sed_prolong']})
    print("prediction accuracy based on test data is %f " % tree.score(test[features], test['sed_prolong']))
    # print(tree.feature_importances_)
    # corr = spearmanr(train['day_of_week'], y)[0]
    # p = spearmanr(train['day_of_week'], y)[1]
    return result_compare


results = predict_sed(sed_train, sed_test)
print(results.head())

from matplotlib import pyplot
# pyplot.plot(results['predicted'])sd
# pyplot.show()

# pyplot.plot(results['value'], color='red')
# pyplot.show()


# get predicted_sed_prolong_start_time
def get_sed_prolong_start_time(data, reference_col, new_col_name):
    # find: 1. starting time of prolonged sedentary behavior, 2. length of each prolonged sedentary time
    data[new_col_name] = 0
    m = 0
    n = 1
    while n < len(data):
        if data.loc[m, reference_col] == 1:
            if data.loc[n, reference_col] == 1:
                # check if consecutive
                t_1 = pd.to_datetime(data.loc[n - 1, 'date'] + ' ' + data.loc[n - 1, 'time'])  # TODO: modify timestamp representation
                t_2 = pd.to_datetime(data.loc[n, 'date'] + ' ' + data.loc[n, 'time'])
                time_delta = t_2 - t_1
                if time_delta == dt.timedelta(minutes=15):
                    data.loc[m, new_col_name] += 1
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
    return data


new_results = get_sed_prolong_start_time(results, 'predicted', 'predicted_sed_prolong_start_time')
print(new_results.head())


def get_notification(predicted, reference_col):
    predicted = copy.deepcopy(predicted)
    predicted = predicted[predicted[reference_col] >= 5]
    predicted = predicted.reset_index()
    date_time = []
    notification = []
    for i in predicted.index:
        num_msg = math.floor((predicted.loc[i, reference_col] + 1) / 6)
        start_time = pd.to_datetime(predicted.date_time[i]) + DateOffset(hours=1.5)
        for j in range(num_msg):
            date_time.append(start_time)
            notification.append("Would you like to take a break?")
            start_time = start_time + DateOffset(hours=1.5)
    notification = pd.DataFrame({'date_time': date_time, 'notification': notification})
    print("notifications generated as follow")
    print(notification)
    return notification

get_notification(new_results, 'sed_prolong_start_time')
get_notification(new_results, 'predicted_sed_prolong_start_time')

