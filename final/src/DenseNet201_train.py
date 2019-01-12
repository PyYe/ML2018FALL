import numpy as np
import os, sys
import random as rn
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU, PReLU, Input
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, TensorBoard, ReduceLROnPlateau
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def create_model(input_shape, n_out):
    
    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = Dense(CHANNEL_NUM)(bn)
    x = Dense(3)(x)
    x = DenseNet201(include_top=False, weights='imagenet', input_shape=(IMAGE_LENGTH, IMAGE_WIDTH, 3), pooling='avg')(x)
    output = Dense(n_out, activation='softmax')(x)
    model = Model(input_tensor, output)
    return model

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

if __name__ == "__main__":
    
    ### Setting
    # Setting gpu
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
    
    ### Initializing arg
    NAME = "test3_notgenerator_densenet201_dense"
    PATH = os.getcwd()
    
    TRAIN = sys.argv[1]    
    TEST = sys.argv[2]    
    PREPROCESSED = sys.argv[3]    
    LABELS = sys.argv[4]    
    SAMPLE = sys.argv[5]    
    MODEL = sys.argv[6]    
    RESULT = sys.argv[7]  
    
    IMAGE_LENGTH = 512
    IMAGE_WIDTH = 512
    CHANNEL_NUM = 4
    TRAIN_SIZE = int(len(os.listdir(TRAIN))/4)
    LABEL_NUM = 28
    
    train_x = np.load(os.path.join(PREPROCESSED, 'train_RGBY_original_x.npy'))
    train_y = np.load(os.path.join(PREPROCESSED, 'train_RGBY_original_y.npy'))
    
    model = create_model(
            input_shape=(IMAGE_LENGTH,IMAGE_WIDTH,CHANNEL_NUM), 
            n_out=LABEL_NUM)
    
    for layer in model.layers:
        layer.trainable = True
    model.layers[-2].trainable = False
    ### Show model summary
    print(model.summary())
    
    adam = Adam(lr=1e-4)
    epochs_to_wait_for_improve = 10
    batch_size = 4
    #valid_split_ratio = 0.2
    n_epochs = 4
    
    cw = np.load(os.path.join(PREPROCESSED, 'class_weight.npy'))
    model.compile(loss=focal_loss(), optimizer=adam, metrics=[f1]) 
    checkpoint_callback = ModelCheckpoint(MODEL, monitor='val_f1'
                                          , verbose=1, save_best_only=True, mode='max')
    
    train_history=model.fit(train_x, train_y
                                      , batch_size = batch_size
                                      , epochs = n_epochs
                                      , validation_split = 0.1
                                    , shuffle=True, class_weight=cw
                                      , callbacks=[    
                                          checkpoint_callback, TensorBoard(log_dir='./tmp/log')
                                                   , ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.0001)
                                                  ]
                                      , verbose=1)
    #!tensorboard --logdir=./tmp/log
    for layer in model.layers:
        layer.trainable = True
    model.layers[-2].trainable = True
    ### Show model summary
    print(model.summary())
    
    n_epochs = 40
    
    model.compile(loss=focal_loss(), optimizer=adam, metrics=[f1]) 
    
    train_history=model.fit(train_x, train_y
                                      , batch_size = batch_size
                                      , epochs = n_epochs
                                      , validation_split = 0.1
                                    , shuffle=True, class_weight=cw
                                      , callbacks=[
                                          checkpoint_callback, TensorBoard(log_dir='./tmp/log')
                                                   , ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.00001)
                                                  ]
                                      , verbose=1)