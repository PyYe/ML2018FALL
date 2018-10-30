
import sys
import numpy as np
import pandas as pd
import math
import os

def sigmoid(z):
    # Prevent overflow.
    #z = np.clip(z, -50, 50)
    return 1/(1+np.exp(-z))

if __name__ == "__main__":
    INPUT_X_PATH = sys.argv[1] # ~/data/train_x.csv
    INPUT_Y_PATH = sys.argv[2] # ~/data/train_y.csv 
    INPUT_TEST_X_PATH = sys.argv[3] # ~/data/test_x.csv
    MODEL_W_PATH = 'model_w.npy' # model_w.npy
    MODEL_minmaxarray_PATH = 'minmaxarray.npy' # 'minmaxarray.npy' for mean normalization
    
    ###produce train_x train_y and train
    df_train_x = pd.read_csv(INPUT_X_PATH)
    df_train_y = pd.read_csv(INPUT_Y_PATH)
    df_test_x = pd.read_csv(INPUT_TEST_X_PATH)
    
    df_train = pd.concat([df_train_x,df_train_y],axis=1)
    df_x = pd.concat([df_train_x, df_test_x], axis = 0)
    
    ### produce minmaxlist minmaxarray
    minmaxlist = []
    minmaxlist.append([min(df_x['LIMIT_BAL']), max(df_x['LIMIT_BAL'])])
    minmaxlist.append([min(df_x['AGE']), max(df_x['AGE'])])
    minmaxlist.append([min(df_x['BILL_AMT1']), max(df_x['BILL_AMT1'])])
    minmaxlist.append([min(df_x['BILL_AMT2']), max(df_x['BILL_AMT2'])])
    minmaxlist.append([min(df_x['BILL_AMT3']), max(df_x['BILL_AMT3'])])
    minmaxlist.append([min(df_x['BILL_AMT4']), max(df_x['BILL_AMT4'])])
    minmaxlist.append([min(df_x['BILL_AMT5']), max(df_x['BILL_AMT5'])])
    minmaxlist.append([min(df_x['BILL_AMT6']), max(df_x['BILL_AMT6'])])
    minmaxlist.append([min(df_x['PAY_AMT1']), max(df_x['PAY_AMT1'])])
    minmaxlist.append([min(df_x['PAY_AMT2']), max(df_x['PAY_AMT2'])])
    minmaxlist.append([min(df_x['PAY_AMT3']), max(df_x['PAY_AMT3'])])
    minmaxlist.append([min(df_x['PAY_AMT4']), max(df_x['PAY_AMT4'])])
    minmaxlist.append([min(df_x['PAY_AMT5']), max(df_x['PAY_AMT5'])])
    minmaxlist.append([min(df_x['PAY_AMT6']), max(df_x['PAY_AMT6'])])
    minmaxarray = np.array(minmaxlist)
    
    df_train_x_onehot = pd.DataFrame(index=df_train_x.index)
    
    ###one-hot encoding
    #LIMIT_BAL
    df_train_x_onehot['LIMIT_BAL'] = df_train_x['LIMIT_BAL']
    #SEX        C
    for i in df_x['SEX'].unique():
        df_train_x_onehot['SEX_'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['SEX'])], index = df_train_x_onehot.index)
    
    
    #EDUCATION  C
    for i in df_x['EDUCATION'].unique():
        df_train_x_onehot['EDUCATION'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['EDUCATION'])], index = df_train_x_onehot.index)
    
    #MARRIAGE   C
    for i in df_x['MARRIAGE'].unique():
        df_train_x_onehot['MARRIAGE'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['MARRIAGE'])], index = df_train_x_onehot.index)
    
    #AGE
    df_train_x_onehot['AGE'] = df_train_x['AGE']
    
    #PAY_0     C    
    for i in df_x['PAY_0'].unique():
        df_train_x_onehot['PAY_0'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['PAY_0'])], index = df_train_x_onehot.index)
    
    #PAY_2     C
    for i in df_x['PAY_2'].unique():
        df_train_x_onehot['PAY_2'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['PAY_2'])], index = df_train_x_onehot.index)
    
    #PAY_3     C
    for i in df_x['PAY_3'].unique():
        df_train_x_onehot['PAY_3'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['PAY_3'])], index = df_train_x_onehot.index)
    
    #PAY_4     C
    for i in df_x['PAY_4'].unique():
        df_train_x_onehot['PAY_4'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['PAY_4'])], index = df_train_x_onehot.index)
    
    #PAY_5     C
    for i in df_x['PAY_5'].unique():
        df_train_x_onehot['PAY_5'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['PAY_5'])], index = df_train_x_onehot.index)
    
    #PAY_6     C
    for i in df_x['PAY_6'].unique():
        df_train_x_onehot['PAY_6'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_train_x['PAY_6'])], index = df_train_x_onehot.index)
    
    #BILL_AMT1 
    df_train_x_onehot['BILL_AMT1'] = df_train_x['BILL_AMT1']
    #BILL_AMT2
    df_train_x_onehot['BILL_AMT2'] = df_train_x['BILL_AMT2']
    #BILL_AMT3
    df_train_x_onehot['BILL_AMT3'] = df_train_x['BILL_AMT3']
    #BILL_AMT4
    df_train_x_onehot['BILL_AMT4'] = df_train_x['BILL_AMT4']
    #BILL_AMT5
    df_train_x_onehot['BILL_AMT5'] = df_train_x['BILL_AMT5']
    #BILL_AMT6
    df_train_x_onehot['BILL_AMT6'] = df_train_x['BILL_AMT6']
    
    #PAY_AMT1
    df_train_x_onehot['PAY_AMT1'] = df_train_x['PAY_AMT1']
    #PAY_AMT2
    df_train_x_onehot['PAY_AMT2'] = df_train_x['PAY_AMT2']
    #PAY_AMT3
    df_train_x_onehot['PAY_AMT3'] = df_train_x['PAY_AMT3']
    #PAY_AMT4
    df_train_x_onehot['PAY_AMT4'] = df_train_x['PAY_AMT4']
    #PAY_AMT5
    df_train_x_onehot['PAY_AMT5'] = df_train_x['PAY_AMT5']
    #PAY_AMT6
    df_train_x_onehot['PAY_AMT6'] = df_train_x['PAY_AMT6']
    
    ###normalize
    #LIMIT_BAL
    df_train_x_onehot['LIMIT_BAL'] = (df_train_x['LIMIT_BAL'] - minmaxarray[0,0] ) / (minmaxarray[0,0]+minmaxarray[0,1])
    
    #AGE
    df_train_x_onehot['AGE'] = (df_train_x['AGE'] - minmaxarray[1,0] ) / (minmaxarray[1,0]+minmaxarray[1,1])
    #BILL_AMT1 
    df_train_x_onehot['BILL_AMT1'] = (df_train_x['BILL_AMT1'] - minmaxarray[2,0] ) / (minmaxarray[2,0]+minmaxarray[2,1])
    #BILL_AMT2
    df_train_x_onehot['BILL_AMT2'] = (df_train_x['BILL_AMT2'] - minmaxarray[3,0] ) / (minmaxarray[3,0]+minmaxarray[3,1])
    #BILL_AMT3
    df_train_x_onehot['BILL_AMT3'] = (df_train_x['BILL_AMT3'] - minmaxarray[4,0] ) / (minmaxarray[4,0]+minmaxarray[4,1])
    #BILL_AMT4
    df_train_x_onehot['BILL_AMT4'] = (df_train_x['BILL_AMT4'] - minmaxarray[5,0] ) / (minmaxarray[5,0]+minmaxarray[5,1])
    #BILL_AMT5
    df_train_x_onehot['BILL_AMT5'] = (df_train_x['BILL_AMT5'] - minmaxarray[6,0] ) / (minmaxarray[6,0]+minmaxarray[6,1])
    #BILL_AMT6
    df_train_x_onehot['BILL_AMT6'] = (df_train_x['BILL_AMT6'] - minmaxarray[7,0] ) / (minmaxarray[7,0]+minmaxarray[7,1])
    
    #PAY_AMT1
    df_train_x_onehot['PAY_AMT1'] = (df_train_x['PAY_AMT1'] - minmaxarray[8,0] ) / (minmaxarray[8,0]+minmaxarray[8,1])
    #PAY_AMT2
    df_train_x_onehot['PAY_AMT2'] = (df_train_x['PAY_AMT2'] - minmaxarray[9,0] ) / (minmaxarray[9,0]+minmaxarray[9,1])
    #PAY_AMT3
    df_train_x_onehot['PAY_AMT3'] = (df_train_x['PAY_AMT3'] - minmaxarray[10,0] ) / (minmaxarray[10,0]+minmaxarray[10,1])
    #PAY_AMT4
    df_train_x_onehot['PAY_AMT4'] = (df_train_x['PAY_AMT4'] - minmaxarray[11,0] ) / (minmaxarray[11,0]+minmaxarray[11,1])
    #PAY_AMT5
    df_train_x_onehot['PAY_AMT5'] = (df_train_x['PAY_AMT5'] - minmaxarray[12,0] ) / (minmaxarray[12,0]+minmaxarray[12,1])
    #PAY_AMT6
    df_train_x_onehot['PAY_AMT6'] = (df_train_x['PAY_AMT6'] - minmaxarray[13,0] ) / (minmaxarray[13,0]+minmaxarray[13,1])
    
    ### produce train_x, train_y
    train_x = df_train_x_onehot.values[:15000, :]
    valid_x = df_train_x_onehot.values[15000:, :]
    
    train_y = df_train_y.values[:15000, :]
    valid_y = df_train_y.values[15000:, :]
    #add bias
    train_x_bias = np.concatenate(((np.zeros(train_x.shape[0])+1).reshape((train_x.shape[0],1)), train_x ), axis=1)
    
    #宣告 weight vector、初始learning rate、# of iteration
    train_w = (np.zeros(train_x_bias.shape[1])).reshape((train_x_bias.shape[1],1))
    s_grad = np.zeros(train_x_bias.shape[1]).reshape((train_x_bias.shape[1],1))
    LearningRate = 10
    NumberofIteration = 3000
    
    for i in range(NumberofIteration):
        y_ = np.dot(train_x_bias, train_w).reshape(train_y.shape)
        L = sigmoid(y_) - train_y
        gra = np.dot(train_x_bias.T, L)
        train_w -= LearningRate * gra/train_x_bias.shape[1]
        print ('iteration: %d | Acc: %f  ' % ( i,((sigmoid(np.dot(train_x_bias,train_w)) >0.5)==train_y).mean()))
    
    np.save(MODEL_W_PATH, train_w)
    np.save(MODEL_minmaxarray_PATH, minmaxarray)