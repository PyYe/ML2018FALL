import sys
import numpy as np
import pandas as pd
import math
import os

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train(w, X, b):
    # Z (Gaussian Distribution).
    z = np.dot(w, X.T) + b
    # Put z into simoid function.
    y = sigmoid(z)
    y_pred = (y.reshape((y.shape[0],1)) > 0.5).astype(np.int)
    return y_pred

if __name__ == "__main__":
    INPUT_PATH = sys.argv[1] # ~/data/test_x.csv
    MODEL_W_PATH = 'model_w.npy' # model_w.npy
    MODEL_b_PATH = 'model_b.npy' # model_b.npy
    OUTPUT_PATH = sys.argv[2] # ~/result/ans.csv
    
    
    df_test_x = pd.read_csv(INPUT_PATH)
    
    w = np.load(MODEL_W_PATH)
    b = np.load(MODEL_b_PATH)
    
    test_y_predict = train(w, df_test_x.values, b)
    
    idlist = []
    for i in range(test_y_predict.shape[0]):
        idlist.append('id_'+str(i))
    
    df_submission = pd.DataFrame({'id':idlist,
                                  'value': test_y_predict.flatten()  })
    
    df_submission.to_csv(OUTPUT_PATH,index=False)