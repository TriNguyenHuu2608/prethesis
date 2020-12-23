import numpy as np
import pandas as pd
import random
import matplotlib.pyplot
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Data_set import Data_set
from sklearn.linear_model import LinearRegression
import sklearn
name="NH3"
q=Data_set(name)
x,y = q.get_data_ga()
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)
print('y_train: ', Y_train)
print('y_test: ', Y_test)
lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test).astype(int)
print('Y_pred: ', Y_pred)


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)

