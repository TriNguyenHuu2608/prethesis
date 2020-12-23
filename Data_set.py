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
    def __init__(self,label):
        self.label = label
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
                if self.label in file:
                    data_concentration = float((file.split(" ")[1]).split('.xlsx')[0]) #Detect Concentration_data ex:1.5 with "H2S 1.5"
                    df = pd.read_excel(f'{root}/{file}')
                    df_row_load = len(df)#-1 if len(df) < 5 else len(df) -2
                    # print('df_row_load: ', df_row_load)
                    x_train = df.iloc[0:df_row_load,100:].values  
                    y_train = np.full(df_row_load,data_concentration)
                    if len(self.X_train) == 0:
                        self.X_train = x_train
                    else:
                        self.X_train = np.vstack([self.X_train,x_train])
                    self.Y_train = np.append(self.Y_train,y_train)
        
        self.Y_train = self.Y_train * 10 #NH3

        return self.X_train,self.Y_train