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
        self.test = np.array([])

    def get_data_ga(self):
        '''
        Get GA data.
        '''
        value_dict =   {'NH3': 10,
                        'NO3' : 3.333,
                        'H2S' : 2}

        for (root,dirs,files) in os.walk ('data_luanvan', topdown= True): #loop through files in folder
            for file in files:
                if self.label in file and '~' not in file:
                    data_concentration = float((file.split(" ")[1]).split('.xlsx')[0]) #Detect Concentration_data ex:1.5 with "H2S 1.5"
                    df = pd.read_excel(f'{root}/{file}')
                    df_row_load = len(df)#-1 if len(df) < 5 else len(df) -2
                    # print('df_row_load: ', df_row_load)
                    x_train = df.iloc[0:df_row_load,50:200].values  
                    y_train = np.full(df_row_load,data_concentration)
                    if len(self.X_train) == 0:
                        self.X_train = x_train
                    else:
                        self.X_train = np.vstack([self.X_train,x_train])
                    self.Y_train = np.append(self.Y_train,y_train)
        
        self.Y_train = (self.Y_train * value_dict.get(self.label))

        return self.X_train,self.Y_train

    def get_data_tf2(self):
        df = pd.read_csv(f'data_combine/{self.label}.csv')
        df_row_load = len(df)
        x_train_1 = df.iloc[0:df_row_load-1,50:200].values
        x_train_2 = df.iloc[0:df_row_load-1,50:200].values
        self.X_train = np.vstack([x_train_1,x_train_2])
        self.X_train = np.reshape(self.X_train, [df_row_load-1, 2, 150])
        self.Y_train = df.iloc[0:df_row_load-1,-1].values
        test_1 = df.iloc[-1,50:200].values
        test_2 = df.iloc[-1,50:200].values
        self.test = np.array([np.array([test_1,test_2])])
        # print('self.X_train: ', len(self.X_train))
        # print('self.test: ', self.test)
        # print('y_train: ', len(self.Y_train))
        # pass
        return self.X_train,self.Y_train,self.test

        
        
if __name__ == "__main__":
    label = 'NO3'
    x=Data_set(label)
    x.get_data_tf2()
