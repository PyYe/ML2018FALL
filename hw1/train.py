# #!/bin/bash
# python3 train.py ./data/train.csv model.npy

import sys
import numpy as np
import pandas as pd
import math

if __name__ == "__main__":
    INPUT_PATH = sys.argv[1] # train.csv
    MODEL_PATH = sys.argv[2] # model.npy
    
    df = pd.read_csv(INPUT_PATH, encoding='ANSI')
    df1 = pd.DataFrame(index= pd.MultiIndex.from_product([list(df['日期'].unique()), list(df.columns[3:])]), 
                   columns=list(df['測項'].unique()))
    del df['測站']

    df.set_index('日期',inplace=True)
    
    for i in df1.index:
        for c in df1.columns:
            df1.loc[i, c] = df[ df['測項']==c ].loc[ i[0], i[1] ]
        
    del df1['RAINFALL']
    df1_float = pd.DataFrame.copy(df1.astype('float'))    
    
    for c in df1_float.columns:
        if c=='PM2.5':
            df1_float[c] = pd.Series([*map(lambda x: np.nan if (x<=0 or x>=200) else x, df1_float[c])], index = df1_float.index)
        elif c=='RAINFALL':
            continue
        else:
            df1_float[c] = pd.Series([*map(lambda x: np.nan if (x<=0) else x, df1_float[c])],index = df1_float.index)
    
    df1_float.reset_index(drop=True,inplace=True)
    
    dflist = []
    for i in range(12):
        dflist.append(pd.DataFrame.copy(df1_float.iloc[i*480:i*480+480,:]).reset_index(drop=True))
        
    train_x = 0
    train_y = 0
    train_x_null = True
    train_y_null = True
    
    for df_no in dflist:
        for i in df_no.index:
            if (i+9) <= len(df_no.index)-1:
                if train_x_null:
                    train_x = df_no.iloc[i:i+9,:].values.reshape((1,len(df_no.iloc[i:i+9,:].values)*len(df_no.iloc[i:i+9,:].values[0])))
                    train_x_null = False
                else:
                    train_x = np.append(train_x, df_no.iloc[i:i+9,:].values.reshape((1,len(df_no.iloc[i:i+9,:].values)*len(df_no.iloc[i:i+9,:].values[0]))), axis=0)
            if i>=9:
                if train_y_null:
                    train_y = np.array(df_no.iloc[i,df_no.columns.get_loc("PM2.5")]).reshape((1,1))
                    train_y_null = False
                else:
                    train_y = np.append(train_y, np.array(df_no.iloc[i,df_no.columns.get_loc("PM2.5")]).reshape((1,1)),axis=0)

    train_x_bias = np.concatenate(((np.zeros(train_x.shape[0])+1).reshape((train_x.shape[0],1)), train_x ), axis=1)
    train = np.concatenate((train_x_bias, train_y), axis=1)
    train_dropna = train[~np.isnan(train).any(axis=1)]
    train_x_bias = train_dropna[:,:-1]
    train_y = train_dropna[:,-1].reshape((train_dropna.shape[0], 1))
    
    #宣告 weight vector、初始learning rate、# of iteration
    train_w = (np.zeros(train_x_bias.shape[1])).reshape((train_x_bias.shape[1],1))
    s_grad = np.zeros(train_x_bias.shape[1]).reshape((train_x_bias.shape[1],1))
    LearningRate = 0.1
    NumberofIteration = 400000
    
    
    for i in range(NumberofIteration):
        y_ = np.dot(train_x_bias, train_w).reshape(train_y.shape)
        L = y_ - train_y
        gra = np.dot(train_x_bias.T, L)
        s_grad += gra**2
        ada = np.sqrt(s_grad)
        train_w -= LearningRate * gra/ada
    
        cost = np.sum(L**2) / train_x_bias.shape[0]
        cost_a  = math.sqrt(cost)
        #print ('iteration: %d | Cost: %f  ' % ( i,cost_a))    
    

    
    np.save(MODEL_PATH, train_w)