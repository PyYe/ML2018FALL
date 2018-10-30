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
    INPUT_PATH = sys.argv[1] # ~/data/test_x.csv
    OUTPUT_PATH = sys.argv[2] # ~/result/ans.csv
    INPUT_X_PATH = sys.argv[3] # ~/data/train_x.csv
    MODEL_W_PATH = 'model_w_best.npy' # model_w.npy
    MODEL_minmaxarray_PATH = 'minmaxarray_best.npy' # 'minmaxarray.npy' for mean normalization
    
    df_test_x = pd.read_csv(INPUT_PATH)
    train_w = np.load(MODEL_W_PATH)
    minmaxarray = np.load(MODEL_minmaxarray_PATH)
    df_train_x = pd.read_csv(INPUT_X_PATH)
    df_x = pd.concat([df_train_x, df_test_x], axis = 0)
    df_test_x_onehot = pd.DataFrame(index=df_test_x.index)
    
    ###one-hot encoding
    #LIMIT_BAL
    df_test_x_onehot['LIMIT_BAL'] = df_test_x['LIMIT_BAL']
    #SEX        C
    for i in df_x['SEX'].unique():
        df_test_x_onehot['SEX_'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['SEX'])], index = df_test_x_onehot.index)
    
    
    #EDUCATION  C
    for i in df_x['EDUCATION'].unique():
        df_test_x_onehot['EDUCATION'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['EDUCATION'])], index = df_test_x_onehot.index)
    
    #MARRIAGE   C
    for i in df_x['MARRIAGE'].unique():
        df_test_x_onehot['MARRIAGE'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['MARRIAGE'])], index = df_test_x_onehot.index)
    
    #AGE
    df_test_x_onehot['AGE'] = df_test_x['AGE']
    
    #PAY_0     C    
    for i in df_x['PAY_0'].unique():
        df_test_x_onehot['PAY_0'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['PAY_0'])], index = df_test_x_onehot.index)
    
    #PAY_2     C
    for i in df_x['PAY_2'].unique():
        df_test_x_onehot['PAY_2'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['PAY_2'])], index = df_test_x_onehot.index)
    
    #PAY_3     C
    for i in df_x['PAY_3'].unique():
        df_test_x_onehot['PAY_3'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['PAY_3'])], index = df_test_x_onehot.index)
    
    #PAY_4     C
    for i in df_x['PAY_4'].unique():
        df_test_x_onehot['PAY_4'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['PAY_4'])], index = df_test_x_onehot.index)
    
    #PAY_5     C
    for i in df_x['PAY_5'].unique():
        df_test_x_onehot['PAY_5'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['PAY_5'])], index = df_test_x_onehot.index)
    
    #PAY_6     C
    for i in df_x['PAY_6'].unique():
        df_test_x_onehot['PAY_6'+str(i)] = pd.Series([*map(lambda x: 1 if str(x)==str(i) else 0, df_test_x['PAY_6'])], index = df_test_x_onehot.index)
    
    #BILL_AMT1 
    df_test_x_onehot['BILL_AMT1'] = df_test_x['BILL_AMT1']
    #BILL_AMT2
    df_test_x_onehot['BILL_AMT2'] = df_test_x['BILL_AMT2']
    #BILL_AMT3
    df_test_x_onehot['BILL_AMT3'] = df_test_x['BILL_AMT3']
    #BILL_AMT4
    df_test_x_onehot['BILL_AMT4'] = df_test_x['BILL_AMT4']
    #BILL_AMT5
    df_test_x_onehot['BILL_AMT5'] = df_test_x['BILL_AMT5']
    #BILL_AMT6
    df_test_x_onehot['BILL_AMT6'] = df_test_x['BILL_AMT6']
    
    #PAY_AMT1
    df_test_x_onehot['PAY_AMT1'] = df_test_x['PAY_AMT1']
    #PAY_AMT2
    df_test_x_onehot['PAY_AMT2'] = df_test_x['PAY_AMT2']
    #PAY_AMT3
    df_test_x_onehot['PAY_AMT3'] = df_test_x['PAY_AMT3']
    #PAY_AMT4
    df_test_x_onehot['PAY_AMT4'] = df_test_x['PAY_AMT4']
    #PAY_AMT5
    df_test_x_onehot['PAY_AMT5'] = df_test_x['PAY_AMT5']
    #PAY_AMT6
    df_test_x_onehot['PAY_AMT6'] = df_test_x['PAY_AMT6']
    
    
    ###normalize
    #LIMIT_BAL
    df_test_x_onehot['LIMIT_BAL'] = (df_test_x['LIMIT_BAL'] - minmaxarray[0,0] ) / (minmaxarray[0,0]+minmaxarray[0,1])
    
    #AGE
    df_test_x_onehot['AGE'] = (df_test_x['AGE'] - minmaxarray[1,0] ) / (minmaxarray[1,0]+minmaxarray[1,1])
    #BILL_AMT1 
    df_test_x_onehot['BILL_AMT1'] = (df_test_x['BILL_AMT1'] - minmaxarray[2,0] ) / (minmaxarray[2,0]+minmaxarray[2,1])
    #BILL_AMT2
    df_test_x_onehot['BILL_AMT2'] = (df_test_x['BILL_AMT2'] - minmaxarray[3,0] ) / (minmaxarray[3,0]+minmaxarray[3,1])
    #BILL_AMT3
    df_test_x_onehot['BILL_AMT3'] = (df_test_x['BILL_AMT3'] - minmaxarray[4,0] ) / (minmaxarray[4,0]+minmaxarray[4,1])
    #BILL_AMT4
    df_test_x_onehot['BILL_AMT4'] = (df_test_x['BILL_AMT4'] - minmaxarray[5,0] ) / (minmaxarray[5,0]+minmaxarray[5,1])
    #BILL_AMT5
    df_test_x_onehot['BILL_AMT5'] = (df_test_x['BILL_AMT5'] - minmaxarray[6,0] ) / (minmaxarray[6,0]+minmaxarray[6,1])
    #BILL_AMT6
    df_test_x_onehot['BILL_AMT6'] = (df_test_x['BILL_AMT6'] - minmaxarray[7,0] ) / (minmaxarray[7,0]+minmaxarray[7,1])
    
    #PAY_AMT1
    df_test_x_onehot['PAY_AMT1'] = (df_test_x['PAY_AMT1'] - minmaxarray[8,0] ) / (minmaxarray[8,0]+minmaxarray[8,1])
    #PAY_AMT2
    df_test_x_onehot['PAY_AMT2'] = (df_test_x['PAY_AMT2'] - minmaxarray[9,0] ) / (minmaxarray[9,0]+minmaxarray[9,1])
    #PAY_AMT3
    df_test_x_onehot['PAY_AMT3'] = (df_test_x['PAY_AMT3'] - minmaxarray[10,0] ) / (minmaxarray[10,0]+minmaxarray[10,1])
    #PAY_AMT4
    df_test_x_onehot['PAY_AMT4'] = (df_test_x['PAY_AMT4'] - minmaxarray[11,0] ) / (minmaxarray[11,0]+minmaxarray[11,1])
    #PAY_AMT5
    df_test_x_onehot['PAY_AMT5'] = (df_test_x['PAY_AMT5'] - minmaxarray[12,0] ) / (minmaxarray[12,0]+minmaxarray[12,1])
    #PAY_AMT6
    df_test_x_onehot['PAY_AMT6'] = (df_test_x['PAY_AMT6'] - minmaxarray[13,0] ) / (minmaxarray[13,0]+minmaxarray[13,1])
    
    
    ### produce test_x
    test_x = df_test_x_onehot.values
    
    ### add bias
    test_x_bias = np.concatenate(((np.zeros(test_x.shape[0])+1).reshape((test_x.shape[0],1)), test_x), axis=1)
    
    ###predict
    test_y_predict = (sigmoid(np.dot(test_x_bias,train_w)) >0.5).astype(np.int)
    
    
    ### output ans.csv
    idlist = []
    for i in range(test_y_predict.shape[0]):
        idlist.append('id_'+str(i))
    
    df_submission = pd.DataFrame({'id':idlist,
                                  'value': test_y_predict.flatten()  })
    
    df_submission.to_csv(OUTPUT_PATH, index=False)