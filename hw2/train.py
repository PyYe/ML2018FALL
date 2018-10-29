import sys
import numpy as np
import pandas as pd
import math
import os

def sigmoid(z):
    return 1/(1+np.exp(-z))

def wb(N_0, N_1, mu_0, mu_1, sigma):
    # Inverse of shared sigma.
    sigma_inv = np.linalg.inv(sigma)
    # Weight.
    w = np.dot((mu_0 - mu_1).T, sigma_inv)
    # Bias.
    b = -0.5 * np.dot(np.dot(mu_0.T, sigma_inv), mu_0) + 0.5 * np.dot(np.dot(mu_1.T, sigma_inv), mu_1) + np.log(float(N_0) / float(N_1))
    return w, b

def train(w, X, b):
    # Z (Gaussian Distribution).
    z = np.dot(w, X.T) + b
    # Put z into simoid function.
    y = sigmoid(z)
    y_pred = (y > 0.5).astype(np.int)
    return y_pred

if __name__ == "__main__":
    INPUT_X_PATH = sys.argv[1] # ~/data/train_x.csv
    INPUT_Y_PATH = sys.argv[2] # ~/data/train_y.csv 
    MODEL_W_PATH = 'model_w.npy' # model_w.npy
    MODEL_b_PATH = 'model_b.npy' # model_b.npy
    
    ###produce train_x train_y and train
    df_train_x = pd.read_csv(INPUT_X_PATH)
    df_train_y = pd.read_csv(INPUT_Y_PATH)
    df_train = pd.concat([df_train_x,df_train_y],axis=1)
    
    ### split train into class0 and class1
    #class0
    df_train_class0 = pd.DataFrame.copy(df_train[df_train['Y'] == 0])
    df_train_class0_x = pd.DataFrame.copy(df_train_class0)
    del df_train_class0_x['Y']
    #class1
    df_train_class1 = pd.DataFrame.copy(df_train[df_train['Y'] == 1])
    df_train_class1_x = pd.DataFrame.copy(df_train_class1)
    del df_train_class1_x['Y']
    
    
    ###calculate mu
    #mu_0
    mu_0 = (np.sum(df_train_class0_x.values, axis=0)/len(df_train_class0_x.index)).reshape((df_train_class0_x.values.shape[1], 1))
    #mu_1
    mu_1 = (np.sum(df_train_class1_x.values, axis=0)/len(df_train_class1_x.index)).reshape((df_train_class1_x.values.shape[1], 1))
    
    ###calculate sigma
    #sigma_0
    s_0 = np.zeros( (mu_0.shape[0], mu_0.shape[0]) )
    for x_0 in df_train_class0_x.values:
        x_0 = x_0.reshape((df_train_class0_x.values.shape[1], 1))
        s_0 += np.dot(x_0-mu_0, (x_0-mu_0).T)

    sigma_0 = s_0/len(df_train_class0_x.index)
    #sigma_1 = np.sum(df_train_class1_x.values, axis=0)/len(df_train_class1_x.index)
    s_1 = np.zeros( (mu_1.shape[0], mu_1.shape[0]) )
    for x_1 in df_train_class1_x.values:
        x_1 = x_1.reshape((df_train_class1_x.values.shape[1], 1))
        s_1 += np.dot(x_1-mu_1, (x_1-mu_1).T)
    
    sigma_1 = s_1/len(df_train_class1_x.index)
    
    ###calculate  shared sigma
    #sigma = (class0 / class0+class1)sigma_0 + (class1 / class0+class1)sigma_1
    sigma = (len(df_train_class1_x.index) / len(df_train_class0_x.index)+len(df_train_class1_x.index))*sigma_0 + \
            (len(df_train_class1_x.index) / len(df_train_class0_x.index)+len(df_train_class1_x.index))*sigma_1
    
    ### calculate w, b
    w, b = wb(len(df_train_class0_x.index), len(df_train_class1_x.index), mu_0, mu_1, sigma)
    
    np.save(MODEL_W_PATH, w)
    np.save(MODEL_b_PATH, b)