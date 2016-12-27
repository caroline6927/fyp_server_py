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

user = User(21, 'MVPA', 600, 60, 180)

user.sunday_vpa = True
print(user.sunday_vpa)

print(list(range(7, 7)))

import pandas as pd
print(pd.date_range('2016-03-01', periods=0, freq='D'))

a = pd.Series([])
print(a.sample(5))

import math

math.floor(5/len([]))