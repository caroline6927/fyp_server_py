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
sed_test = sed_train[-792:]
sed_test = sed_test.reset_index()
sed_train = sed_train[:2376]


def predict_sed(train, test):
    features = ['1', '2', '3', '4', '5', '6', '7']
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
    result_compare = pd.DataFrame({'predicted': y_predicted, 'value': test['sed_prolong']})
    print(tree.score(test[features], test['sed_prolong']))
    # print(tree.feature_importances_)
    # corr = spearmanr(train['day_of_week'], y)[0]
    # p = spearmanr(train['day_of_week'], y)[1]
    return result_compare



results = predict_sed(sed_train, sed_test)

results_sed = results[results['value'] > 0]
print(results_sed)
score = sum(results_sed['predicted'] == results_sed['value']) / len(results_sed)
print(score)