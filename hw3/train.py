
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
    TRAIN_PATH = sys.argv[1]
    
    train_df = pd.read_csv(TRAIN_PATH)
    
    # Image Informations
    Image_nums = int(len(train_df.index))
    Image_rows = int(len(train_df['feature'][0].split(' '))**(1/2))
    Image_cols = int(len(train_df['feature'][0].split(' '))**(1/2))
    channels = 1 # 1 means grey level ; 3 means color

    label_nums = int(len(train_df['label'].unique()))
    
    train_x_values = train_df['feature'].values

    train_x = np.zeros((Image_nums, Image_rows * Image_cols))
    for n in range(Image_nums):
        train_x[n] = train_x_values[n].split(' ')
    
    ### Reshape 
    ## For 2D data (e.g. image), 
    ## "channels_last" assumes (rows, cols, channels) 
    ## "channels_first" assumes  (channels, rows, cols)
    if K.image_data_format() == 'channels_first':
        train_x = train_x.reshape((Image_nums, channels, Image_rows, Image_cols)).astype('float32')
    else:
        train_x = train_x.reshape((Image_nums, Image_rows, Image_cols, channels)).astype('float32')
        
    ### Image Normalization
    train_x_normalize=train_x/255
    
    ### Label preprocessing
    train_y_values = train_df['label'].values
    train_y_onehot = np_utils.to_categorical(train_y_values)
    
    # Use ImageDataGenerator to implement data augmentation. 
    datagen = ImageDataGenerator(
            rotation_range = 40,
            width_shift_range = 0.3,
            height_shift_range = 0.3,
            zoom_range = [0.6, 1.4],
            shear_range = 0.4,
            horizontal_flip = True)
    
    ###Build CNN model
    model=Sequential() 
    # Conv block 1: 64 output filters.
    model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal', input_shape = (Image_rows, Image_cols, channels)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.3))
    # Conv block 2: 128 output filters.
    model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # Conv block 3: 256 output filters.
    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # Conv block 4: 512 output filters.
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # Conv block 5: 512 output filters.
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 1 / 40))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(units = 4096, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 4096, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1000, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(label_nums, activation = 'softmax'))
    
    ### Show model summary
    print(model.summary())
    
    ###
    adam = Adam(lr=1e-4)
    epochs_to_wait_for_improve = 10
    batch_size = 128
    #valid_split_ratio = 0.2
    n_epochs = 27
        
    ### Split train and valid
    X_train = train_x_normalize[:25710,:,:,:]
    X_valid = train_x_normalize[25710:,:,:,:]
    y_train = train_y_onehot[:25710,:]
    y_valid = train_y_onehot[25710:,:]
    
    ### training
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy']) 
    
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=epochs_to_wait_for_improve)
    checkpoint_callback = ModelCheckpoint(model_name+'.h5', monitor='val_acc'
                                          , verbose=1, save_best_only=True, mode='max')
    
    train_history=model.fit_generator(datagen.flow(X_train, y_train
                                                   , batch_size = batch_size)
                                      ,steps_per_epoch = len(X_train)*10 / batch_size
                                      , epochs=n_epochs
                                      , validation_data=(X_valid, y_valid)
                                      , callbacks=[early_stopping_callback, checkpoint_callback]
                                      , verbose=1)