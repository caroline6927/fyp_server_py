"""
This script analyzes training data and data of current week. Generate physical activity plans
Abbreviation:
MVPA: moderate-vigorous physical activity
VPA: vigorous physical activity
"""
from __future__ import print_function
import subprocess
import configparser
import datetime as dt
import math

import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
from scipy.stats.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier, export_graphviz

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
        self.yesterday_vpa = False
        self.today_vpa = False

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


# mark out MVPA intervals
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


# get daily met
def get_daily_met(user_data):
    user_data = user_data.groupby(['week', 'day_of_week'])
    user_data_daily_met = pd.DataFrame(user_data['met'].sum()).reset_index()
    # get accumulated met of each day, starting from Monday
    user_data_daily_met['met_accumulated'] = user_data_daily_met['met']
    i = 0
    while i < len(user_data_daily_met) - 1:
        if user_data_daily_met.loc[i, 'week'] == user_data_daily_met.loc[i + 1, 'week']:
            user_data_daily_met.loc[i + 1, 'met_accumulated'] += user_data_daily_met.loc[i, 'met_accumulated']
        i += 1
    # get days of week not so active
    inactive = []
    for i in range(len(user_data_daily_met)):
        if user_data_daily_met.loc[i, 'met'] == 0:
            inactive.append(1)
        else:
            inactive.append(0)
    user_data_daily_met['inactive'] = inactive
    return user_data_daily_met


# check correlation between day of week and MET
def check_corr(user_data):
    corr = spearmanr(user_data['day_of_week'], user_data['met_accumulated'])[0]
    p = spearmanr(user_data['day_of_week'], user_data['met_accumulated'])[1]
    print('p value is %f' % p)
    print('corr is %f' % corr)
    return corr, p


def get_plan_mode(corr, p):
    if (p <= 0.05) & (abs(corr) > 0):
        print("Statistically significant correlation found between days of week and daily MET.")
        print("Execute patterned plan generator.")
        plan_mode = 'pattern'
    else:
        print("No strong correlation found between days of week and daily MET.")
        print("Execute random plan generator.")
        plan_mode = 'random'
    return plan_mode


def get_completed_met(current):
    """
    Updates every time user syncs data
    :param current:
    :return:
    """
    current = get_activity_intensity(current)
    current_daily_met = get_daily_met(current)
    user.completed = current_daily_met['met'].sum()


get_completed_met(fitbit_current)


def get_met_estimation(historical_data):
    week_num_last = pd.to_datetime(today).week - 1
    met_estimation = historical_data[historical_data['week'] == week_num_last]['met'].sum()
    user.last_week = met_estimation
    user.update_current_goal()


def check_vpa_yesterday_today(intensity_data):  # put in output of get_activity_intensity of current week data
    dayofweek_yesterday = pd.to_datetime(today).dayofweek - 1
    dayofweek_today = pd.to_datetime(today).dayofweek
    try:
        vpa_yesterday = intensity_data.loc[intensity_data['day_of_week'] == dayofweek_yesterday, 'vpa'].sum()
    except KeyError:
        vpa_yesterday = 0
    try:
        vpa_today = intensity_data.loc[intensity_data['day_of_week'] == dayofweek_today, 'vpa'].sum()
    except KeyError:
        vpa_today = 0
    if vpa_yesterday > 0:
        user.yesterday_vpa = True
    else:
        user.yesterday_vpa = False
    if vpa_today > 0:
        user.today_vpa = True
    else:
        user.today_vpa = False


# generate PA plans
def random_plan_generator():
    """
    Updates every time user syncs data
    :param met_to_complete:
    :param choice:
    :return:
    """
    # get user current status
    user.update_current_goal()
    met_to_complete = user.current_goal
    print(met_to_complete)
    choice = user.choice
    yesterday_vpa = user.yesterday_vpa
    today_vpa = user.today_vpa
    # model of randomly assign PA of choice to days left in this week to fulfill MET to complete
    # generate days left in this week
    # for now, use dummy value for today
    dayofweek_today = pd.to_datetime(today).dayofweek
    # today = dt.datetime.today()
    # dayofweek_today = pd.to_datetime(today).dayofweek
    # as the window for notification is 1400 to 1700, no activity will be arrange for the current day after 1700.

    if dt.datetime.today().time().hour >= 17:
        if today_vpa:
            if dayofweek_today == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = pd.to_datetime(today)
            else:
                days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
                days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
        else:  # no vpa so far today
            days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
            days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)

    else:
        if yesterday_vpa:
            if dayofweek_today == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = pd.to_datetime(today)
            else:
                if today_vpa:
                    days_left = pd.Series(list(range(dayofweek_today + 2, 7)))
                    days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
                else:
                    days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
                    days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
        elif today_vpa:
            if dayofweek_today == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = pd.to_datetime(today)
            else:
                days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
                days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
        else:
            days_left = pd.Series(list(range(dayofweek_today, 7)))
            days_left_start_date = pd.to_datetime(today)

    days_left_date_range = pd.date_range(days_left_start_date, periods=len(days_left), freq='D').strftime(
        '%Y-%m-%d')

    if choice == 'MVPA':
        bouts = math.ceil(met_to_complete / (4 * 15))
    elif choice == 'VPA':
        bouts = math.ceil(met_to_complete / (8 * 15))

    if len(days_left) > 0:
        base_bouts = math.floor(bouts / len(days_left))
        extra_bouts = bouts % len(days_left)

        plan = pd.DataFrame({'date': days_left_date_range, 'day_of_week': days_left, 'week': pd.to_datetime(today).week,
                             'choice': choice, 'bouts': base_bouts})
        days_extra_bouts = days_left.sample(extra_bouts, replace=False)  # as extra_bouts must be smaller than days_left
        for i in days_extra_bouts:
            plan.loc[plan['day_of_week'] == i, 'bouts'] += 1
        return plan
    else:
        return None


def pattern_plan_generator(train):  # input daily MET training data
    # get user current status
    user.update_current_goal()
    met_to_complete = user.current_goal
    # TODO: for each day met_to_complete should also add the usual met predicted yet unperformed as usual
    # print(met_to_complete)
    choice = user.choice
    yesterday_vpa = user.yesterday_vpa
    today_vpa = user.today_vpa
    dayofweek_today = pd.to_datetime(today).dayofweek
    # print(train)
    # predict this week's PA trend
    pattern_tree = DecisionTreeClassifier()
    pattern_tree = pattern_tree.fit(np.array(train[['day_of_week']]), train.loc[:, 'inactive'])
    predicted = pd.DataFrame({'day_of_week': list(range(7))})
    predicted['predicted_inactive'] = pattern_tree.predict(predicted[['day_of_week']])
    accuracy = pattern_tree.score(np.array(train[['day_of_week']]), train.loc[:, 'inactive'])
    print(accuracy)
    # print(pattern_tree.predict_proba(predicted[['day_of_week']]))
    # print(pattern_tree.classes_)

    # testing
    def visualize_tree(tree, feature_names):
        """Create tree png using graphviz.

        Args
        ----
        tree -- scikit-learn DecsisionTree.
        feature_names -- list of feature names.
        """
        with open("dt.dot", 'w') as f:
            export_graphviz(tree, out_file=f,
                            feature_names=feature_names)

        command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
        try:
            subprocess.check_call(command)
        except:
            exit("Could not run dot, ie graphviz, to "
                 "produce visualization")

    visualize_tree(pattern_tree, ['day_of_week'])
    # print(pattern_tree.score())

    # generate days for sending notification
    if dt.datetime.today().time().hour >= 17:
        if today_vpa:
            if dayofweek_today == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = pd.to_datetime(today)
            else:
                days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
                days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
        else:  # no vpa so far today
            days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
            days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)

    else:
        if yesterday_vpa:
            if dayofweek_today == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = pd.to_datetime(today)
            else:
                if today_vpa:
                    days_left = pd.Series(list(range(dayofweek_today + 2, 7)))
                    days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
                else:
                    days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
                    days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
        elif today_vpa:
            if dayofweek_today == 6:  # TODO: notify user to sync on Monday morning
                days_left = pd.Series([])
                days_left_start_date = pd.to_datetime(today)
            else:
                days_left = pd.Series(list(range(dayofweek_today + 1, 7)))
                days_left_start_date = pd.to_datetime(today) + DateOffset(days=1)
        else:
            days_left = pd.Series(list(range(dayofweek_today, 7)))
            days_left_start_date = pd.to_datetime(today)

    predicted = predicted.loc[predicted['day_of_week'].isin(days_left)].reset_index()
    predicted.drop('index', axis=1, inplace=True)
    predicted['date'] = pd.date_range(days_left_start_date, periods=len(days_left), freq='D').strftime('%Y-%m-%d')

    # assign PA to days
    if choice == 'MVPA':
        bouts = math.ceil(met_to_complete / (4 * 15))
    elif choice == 'VPA':
        bouts = math.ceil(met_to_complete / (8 * 15))

    predicted['week'] = pd.to_datetime(today).week
    predicted['choice'] = choice
    predicted['bouts'] = 0

    if len(days_left) > 0:
        num_inactive = len(predicted[predicted['predicted_inactive'] == 1])
        base_bouts = math.floor(bouts / num_inactive)
        extra_bouts = bouts % num_inactive
        # assign base bouts to inactive days
        for i in predicted.index:
            if predicted.loc[i, 'predicted_inactive'] == 1:
                predicted.loc[i, 'bouts'] = base_bouts

        # assign extra bouts to all days left, including active days
        # check if the user needs to be more active during weekdays
        # criteria: user did not fulfill the PA goal and tend to have VPA during weekend
        dayofweek_saturday = 5
        dayofweek_sunday = 6
        saturday_vpa = False
        sunday_vpa = False
        if pd.Series(dayofweek_saturday).isin(predicted.loc[predicted['choice'] == 'VPA', 'day_of_week'])[0]:
            saturday_vpa = True
        else:
            print("saturday checked")
        if pd.Series(dayofweek_sunday).isin(predicted.loc[predicted['choice'] == 'VPA', 'day_of_week'])[0]:
            sunday_vpa = True
        else:
            print("sunday checked")
        if (saturday_vpa | sunday_vpa) & (met_to_complete > 0):
            if days_left[0] <= 4:
                weekday_left = list(set(days_left) - set([5, 6]))
                days_extra_bouts = weekday_left.sample(extra_bouts)
            else:
                days_extra_bouts = []
        else:
            days_extra_bouts = days_left.sample(extra_bouts)

        for i in days_extra_bouts:
            predicted.loc[predicted['day_of_week'] == i, 'bouts'] += 1
        print("print predicted activity")
        print(predicted)
        predicted.to_csv(config.get('default', 'data_for_analysis') + '_presentation_pa_predicted.csv')
        return predicted
    else:
        return None


def get_plan_notification(plan):
    notification = []
    for i in plan.index:
        minutes = plan.loc[i, 'bouts'] * 15
        if plan.loc[i, 'bouts'] > 0:
            if plan.loc[i, 'choice'] == 'MVPA':
                message = "How about a " + str(minutes) + " minutes brisk walking later?"
                notification.append(message)
            if plan.loc[i, 'choice'] == 'VPA':
                message = "How about a " + str(minutes) + " minutes jogging later?"
                notification.append(message)
        else:
            notification.append('None')
    plan['notification'] = notification
    print(plan)
    return plan


# training data
# get activity intensity
fitbit_train = get_activity_intensity(fitbit_train)
# print(fitbit_train.head())
# fitbit_train.to_csv(config.get('default', 'data_for_analysis') + '_presentation_mvpa.csv', encoding='ISO-8859-1')
fitbit_current = get_activity_intensity(fitbit_current)
# get daily MET data
train_daily_met = get_daily_met(fitbit_train)
train_daily_met.to_csv(config.get('default', 'data_for_analysis') + '_presentation_daily_met.csv', encoding='ISO-8859-1')
# print(train_daily_met)
# update user PA profile
get_met_estimation(train_daily_met)
check_vpa_yesterday_today(fitbit_current)

# generate PA notification plan
# check correlation using daily MET data
train_corr, train_p = check_corr(train_daily_met)
activity_plan_mode = get_plan_mode(train_corr, train_p)
# print(random_plan_generator())
if activity_plan_mode == 'random':
    user_plan = random_plan_generator()
if activity_plan_mode == 'pattern':
    user_plan = pattern_plan_generator(train_daily_met)


def output_json_notification(plan):
    plan = plan.set_index(plan['date'])
    try:
        plan.drop(['predicted_inactive', 'date', 'week', 'day_of_week', 'bouts'], axis=1, inplace=True)
    except ValueError:
        plan.drop(['date', 'week', 'day_of_week', 'bouts'], axis=1, inplace=True)
    file_name = config.get('default', 'fitbit_output') + user_id + "_" + today + '_activity_notification.json'
    plan.to_json(file_name)



# output a json file
# get notification
if user_plan is not None:  # user_plan is not None
    user_plan = get_plan_notification(user_plan)
    output_json_notification(user_plan)
else:
    user_plan = None
    print("No more activity plan for this week")

user_plan.to_csv(config.get('default', 'data_for_analysis') + '_presentation_notification.csv')

"""
# testing data
fitbit_test_filename = fitbit_user_data_dir + user_id + '_fitbit_test.csv'
fitbit_test = pd.read_csv(fitbit_train_filename, index_col='time_of_day', header=0, encoding='ISO-8859-1')
# get activity intensity
fitbit_test = get_activity_intensity(fitbit_test)
# get daily MET data
test_daily_met = get_daily_met(fitbit_test)
# check correlation using daily MET data
test_corr, test_p = check_corr(test_daily_met)
"""
