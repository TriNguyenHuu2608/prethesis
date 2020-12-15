import numpy as np
import pandas as pd
from sklearn import preprocessing
import feature_selection
def get_data(filename, label):
    data = pd.read_csv(filename)
    a = np.arange(54, 321)
    b = np.arange(321, 588)
    X1 = data.iloc[5:85, a].values
    Z = data.iloc[200, a].values
    X2 = data.iloc[5:85, b].values
    K = data.iloc[200, b].values
    y = np.full((1, 80), label)
    X_1 = Z/X1
    X_2 = K/X2
    X = np.vstack([X_1, X_2])
    X = np.reshape(X, [80, 2, 267])
    return X, y
def get_mix_data(filename):
    a = np.arange(54, 321)
    b = np.arange(321, 588)
    data = pd.read_csv(filename)
    X1 = data.iloc[11:21, a].values
    Z = data.iloc[300, a].values
    X2 = data.iloc[11:21, b].values
    K = data.iloc[300, b].values
    X_1 = Z/X1
    X_2 = K/X2
    X = np.vstack([X_1, X_2])
    X = np.reshape(X, [10, 2, 267])
    return X
def get_full_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[1:81, 54:588].values
    y = np.full((1, 80), label)
    Z = data.iloc[200, 54:588].values
    X = Z/X
    return X, y
def get_full_mix_data(filename, label):
    data = pd.read_csv(filename)
    X = data.iloc[1:81, 54:588].values
    y = np.full((1, 80), label)
    Z = data.iloc[320, 54:588].values
    X = Z/X
    return X, y
def get_data_Vodka(filename, label):
    data = pd.read_csv(filename)
    a, b = feature_selection.feature_importance()
    X_1 = data.iloc[1:81,a].values
    Z = data.iloc[260,a].values
    X_2 = data.iloc[1:81, b].values
    K = data.iloc[260, b].values
    y = np.full((1, 80), label)
    X_1 = Z/X_1
    X_2 = K/X_2
    X = np.vstack([X_1, X_2])
    X = np.reshape(X, [80, 2, 267])
    return X, y
    import numpy as np
import pandas as pd
import os
import re
from sklearn import preprocessing

class Data_set:
    '''
    Class use to get train and test data.\n
    Param : Name of substance.
    '''
    def __init__(self,name):
        self.name = name
        self.X_train = np.array([])
        self.Y_train = np.array([])
        self.X_test = np.array([])
        self.Y_test = np.array([])

    def get_data_ga(self):
        '''
        Get GA data.
        '''
        for (root,dirs,files) in os.walk ('data_luanvan', topdown= True): #loop through files in folder
            for file in files:
                if self.name in file:
                    data_concentration = float((file.split(" ")[1]).split('.xlsx')[0]) #Detect Concentration_data ex:1.5 with "H2S 1.5"
                    df = pd.read_excel(f'{root}/{file}')
                    df_row_load = len(df)-1 if len(df) < 5 else len(df) -2
                    x_train = df.iloc[0:df_row_load,:200].values  
                    y_train = np.full(df_row_load,data_concentration)
                    if len(self.X_train) == 0:
                        self.X_train = x_train
                    else:
                        self.X_train = np.vstack([self.X_train,x_train])
                    self.Y_train = np.append(self.Y_train,y_train)
        
        return self.X_train,self.Y_train

