import tensorflow as tf
import random as rn
import os
import pandas as pd
import numpy as np
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU, PReLU, Input
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
#keras.layers.advanced_activations
#### Test file
from keras.models import load_model
if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = '0'
    # Setting the seed for numpy-generated random numbers
    np.random.seed(37)
    # Setting the seed for python random numbers
    rn.seed(1254)
    # Setting the graph-level random seed.
    tf.set_random_seed(89)
    # 自動增長 GPU 記憶體用量
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # 設定 Keras 使用的 Session
    tf.keras.backend.set_session(sess)
    
    model_name = 'test15'
    TEST_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    
    ###Open test.csv
    test_df = pd.read_csv(TEST_PATH)
    
    # Fix random seed for reproducibility.
    seed = 101
    np.random.seed(seed)
    
    # Image Informations
    Image_nums_test = int(len(test_df.index))
    Image_rows_test = int(len(test_df['feature'][0].split(' '))**(1/2))
    Image_cols_test = int(len(test_df['feature'][0].split(' '))**(1/2))
    channels_test = 1 # 1 means grey level ; 3 means color
    
    #label_nums = int(len(train_df['label'].unique()))
    
    test_x_values = test_df['feature'].values

    test_x = np.zeros((Image_nums_test, Image_rows_test * Image_cols_test))
    for n in range(Image_nums_test):
        test_x[n] = test_x_values[n].split(' ')
    
    if K.image_data_format() == 'channels_first':
        test_x = test_x.reshape((Image_nums_test, channels_test, Image_rows_test, Image_cols_test)).astype('float32')
    else:
        test_x = test_x.reshape((Image_nums_test, Image_rows_test, Image_cols_test, channels_test)).astype('float32')
    
    ### Image Normalization
    test_x_normalize=test_x/255
    
    ### Load model
    modellist = []
    modelnamelist = [5,11,12,13]
    for i in modelnamelist:
        modellist.append(load_model('test'+str(i)+'.h5'))
    
    test_ys = []
    for m in modellist:
        # # Predict y.
        test_ys.append(m.predict(test_x_normalize, verbose=1))
        
    test_ys = np.array(test_ys)
    test_y_sum = np.sum(test_ys, axis=0)
    test_y_predict = np.argmax(test_y_sum, axis=1)
    
    idlist = []
    for i in range(test_y_predict.shape[0]):
        idlist.append(i)
        
    df_submission = pd.DataFrame({'id':idlist,
                                  'label': test_y_predict  })
    
    df_submission.to_csv(RESULT_PATH,index=False)