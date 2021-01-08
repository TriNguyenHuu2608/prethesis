import numpy as np
import pandas as pd
import random
import sklearn
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Data_set import Data_set
from sklearn.linear_model import LinearRegression
# name="NO3"
# q=Data_set(name)
# x,y = q.get_data_ga()
# print('x: ', x)

# print('y: ', y)
# X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# print('y_train: ', Y_train)
# print('y_test: ', Y_test)
# lm = LinearRegression()
# lm.fit(X_train, Y_train)
# Y_pred = lm.predict(X_test)
# print('Y_pred: ', Y_pred)


# mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
# msez_v2 = abs(Y_test - Y_pred)

# print(mse)
# print('msez_v2: ', msez_v2)
label ="NO3"
data_class = Data_set(label)
x,y,test = data_class.get_data_tf2()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('X_train: ', X_train)
print('y_train: ', y_train)