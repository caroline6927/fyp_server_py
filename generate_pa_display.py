"""
This script gets step, mpa, vpa of the past 7 days including today
"""
import configparser

import pandas as pd
from pandas.tseries.offsets import *

config = configparser.ConfigParser()
config.read('user.ini')
user_id = config.get('default', 'user_id')
fitbit_user_data_dir = config.get('default', 'data_for_analysis')
today = config.get('default', 'today')

# get training data
fitbit_train_filename = fitbit_user_data_dir + user_id + '_fitbit_train.csv'
fitbit_train = pd.read_csv(fitbit_train_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')
# get data of the current week
fitbit_current_filename = fitbit_user_data_dir + user_id + '_fitbit_current.csv'
fitbit_current = pd.read_csv(fitbit_current_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')


class User:
    def __init__(self, age, choice, goal, completed, last_week):
        self.age = age
        self.choice = choice
        self.max_hr = 220 - self.age
        self.mvpa_low = 0.64 * self.max_hr
        self.mvpa_high = 0.77 * self.max_hr
        self.vpa_low = 0.77 * self.max_hr
        self.vpa_high = 0.89 * self.max_hr
        self.goal = goal
        self.completed = completed
        self.last_week = last_week
        self.current_goal = self.goal - self.completed - self.last_week
        self.saturday_vpa = False
        self.sunday_vpa = False
        self.yesterday_vpa = False

    def update_current_goal(self):
        self.current_goal = self.goal - self.completed - self.last_week
        # current goal should be based on goal from previous week minus met completed this week


# specify user settings
def get_user_profile():
    age = config.get('default', 'user_age')
    choice = config.get('default', 'user_choice')
    goal = config.get('default', 'user_goal')
    user_profile = User(int(age), choice, int(goal), 0, 0)
    return user_profile


user = get_user_profile()


def get_activity_intensity(user_data, user_settings=user):
    # process mvpa
    mvpa_select = (user_data.heart >= user_settings.mvpa_low) & (user_data.heart < user_settings.mvpa_high) & (
        user_data.step > 0)
    user_data['mvpa_select'] = mvpa_select
    user_data['mvpa'] = 0
    user_data.loc[user_data.mvpa_select, 'mvpa'] = 1
    user_data.drop('mvpa_select', axis=1, inplace=True)

    # process vpa
    vpa_select = (user_data.heart >= user_settings.vpa_low) & (user_data.heart < user_settings.vpa_high) & (
        user_data.step > 0)
    user_data['vpa_select'] = vpa_select
    user_data['vpa'] = 0
    user_data.loc[user_data.vpa_select, 'vpa'] = 1
    user_data.drop('vpa_select', axis=1, inplace=True)

    # calculate MET
    user_data['met'] = user_data['mvpa'] * 4 * 15 + user_data['vpa'] * 8 * 15

    # drop na rows
    user_data = user_data[(user_data.heart > 0) & (user_data.step > 0)]
    return user_data


def get_daily_activity(user_data):
    date_list = user_data['date'].unique()
    user_data = user_data.groupby(['week', 'day_of_week'])
    user_data_daily_mvpa = pd.DataFrame(user_data['mvpa'].sum() * 15).reset_index()
    user_data_daily_mvpa['date'] = date_list
    user_data_daily_vpa = pd.DataFrame(user_data['vpa'].sum() * 15).reset_index()
    user_data_daily_step = pd.DataFrame(user_data['step'].sum()).reset_index()
    user_data = user_data_daily_mvpa
    user_data['vpa'] = user_data_daily_vpa['vpa']
    user_data['step'] = user_data_daily_step['step']
    return user_data


def get_past_7_days(train, current):
    user_data = pd.concat([train, current])
    start_date = pd.to_datetime(today) - DateOffset(days=6)
    date_range = pd.date_range(start_date, periods=7, freq='D').strftime('%Y-%m-%d')
    user_data = user_data.loc[user_data['date'].isin(date_range)].reset_index()
    date_count = len(user_data) - 1
    user_data['date_count'] = 0
    for i in range(len(user_data)):
        user_data.loc[i, 'date_count'] = date_count
        date_count -= 1
    user_data = user_data.set_index(user_data['date_count'])
    user_data.drop(['week', 'day_of_week', 'index', 'date', 'date_count'], axis=1, inplace=True)
    print(user_data)
    return user_data


train = get_activity_intensity(fitbit_train)
current = get_activity_intensity(fitbit_current)
train = get_daily_activity(train)
current = get_daily_activity(current)

past_7_days = get_past_7_days(train, current)

# output a json file
file_name = config.get('default', 'fitbit_output') + user_id + "_" + today + '_activity_display.json'
past_7_days.to_json(file_name)




