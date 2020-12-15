
# Code source: Jaques Grobler
# License: BSD 3 clause


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# # Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print('Mean squared error: %.2f'
#       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(diabetes_y_test, diabetes_y_pred))

# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
import numpy as np
import pandas as pd
import os
import re
data_inputs = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])



# df = pd.read_excel("data_luanvan/H2S 1.5.xlsx")
# print('df: ', df)
# x=df.iloc[0:2,:10].values
# print('x: ', x)
# y = np.full(3, 1.5)
# print('y: ', y)
# df2 = pd.read_excel("data_luanvan/NH3 0.01.xlsx")
# print('df2: ', df2)
# a=df.iloc[0:2,:10].values
# print('a: ', a)
# b = np.full(3, 1.5)
# print('b: ', b)
# X_train=np.vstack([x,a])
# print('X_train: ', X_train)
# y=np.array([])
# print('leny: ', len(y))
# Y_train = np.append(y,b)
# print('Y_train: ', Y_train)

from Data_set import Data_set
a='NH3'
q = Data_set(a)
print(q.name)
x,y = q.get_data_ga()
print('y: ', y)
print('x: ', len(x))