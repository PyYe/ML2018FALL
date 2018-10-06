# #!/bin/bash
# python3 test.py ./data/test.csv ./result/sampleSubmission.csv


import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    INPUT_PATH = sys.argv[1] # train.csv
    MODEL_PATH = 'model.npy'
    OUTPUT_PATH = sys.argv[2] # model.npy
    
    
    df_test = pd.read_csv(INPUT_PATH, encoding='ANSI', header=None)
    
    df_test1 = pd.DataFrame(index= pd.MultiIndex.from_product([list(df_test[0].unique()), list(df_test.columns[2:])]), 
                   columns=list(df_test[1].unique()))
    
    df_test.set_index(0,inplace=True)
    df_test.replace('NR',0,inplace=True)
    
    for i in df_test1.index:
        for c in df_test1.columns:
            df_test1.loc[i, c] = df_test[ df_test[1]==c ].loc[ i[0], i[1] ]
    
    del df_test1['RAINFALL']
    df_test1 = df_test1.astype('float')
    
    for c in df_test1.columns:
        if c=='PM2.5':
            df_test1[c] = pd.Series([*map(lambda x: np.nan if (x<=0 or x>=200) else x, df_test1[c])], index = df_test1.index)
        elif c=='RAINFALL':
            continue
        else:
            df_test1[c] = pd.Series([*map(lambda x: np.nan if (x<=0) else x, df_test1[c])],index = df_test1.index)
            
    #linearly interpolate

    for c in df_test1.columns:
        df_test1[c].interpolate(inplace = True)

    test_x_null = True
    index0 = '0'
    for i in df_test1.index:
        if i[0] != index0:
            index0 = i[0]
            if test_x_null:
                test_x = df_test1.loc[i[0], :].values.flatten().reshape((1,df_test1.loc[i[0], :].values.flatten().shape[0]))
                test_x_null = False
            else:
                test_x = np.append(test_x, df_test1.loc[i[0], :].values.flatten().reshape((1,df_test1.loc[i[0], :].values.flatten().shape[0])),axis=0)
    
    test_x_bias = np.concatenate(((np.zeros(test_x.shape[0])+1).reshape((test_x.shape[0],1)), test_x), axis=1)
    
    test_x_bias = test_x_bias.astype('float')
    
    
    train_w = np.load(MODEL_PATH)
    
    idlist = []
    for i in range(np.dot(test_x_bias,train_w).shape[0]):
        idlist.append('id_'+str(i))
    
    df_submission = pd.DataFrame({'id':idlist,
                                  'value': np.dot(test_x_bias,train_w).flatten()  })
    
    df_submission.to_csv(OUTPUT_PATH, index=False)