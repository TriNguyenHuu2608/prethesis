import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Data_set import Data_set
import Show_result
import Train_tf2
def main(argv = None):   
    label ="NO3"
    data_class = Data_set(label)
    x,y,test = data_class.get_data_tf2()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    loss, val_loss = Train_tf2.train(X_train, X_test, y_train, y_test, X_val, y_val, test)
    num_epochs = 2000
    Show_result.show_result_tf2(loss, val_loss, num_epochs)
if __name__ == '__main__':
    tf.compat.v1.app.run()
    # label ="NO3"
    # data_class = Data_set(label)
    # x,y,test = data_class.get_data_tf2()
    # print('x: ', x)
    # print('y: ', y)
    # print('test: ', test)






