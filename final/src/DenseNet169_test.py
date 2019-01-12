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
from keras.applications.densenet import DenseNet201, DenseNet169

from tta_wrapper import tta_classification

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
    x = DenseNet169(include_top=False, weights='imagenet', input_shape=(IMAGE_LENGTH, IMAGE_WIDTH, 3), pooling='avg')(x)
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
    NAME = "test5_notgenerator_densenet169_dense_TTA_test"
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
    LABEL_NUM = 28
    
    test_x = np.load(os.path.join(PREPROCESSED, 'test_RGBY_original_x.npy'))
    
    model = load_model(MODEL, custom_objects={'f1': f1, 'focal_loss_fixed' : focal_loss()})
    
    tta_model = tta_classification(model, h_flip=True, rotation=(90, 270), 
                             merge='mean')


    test_pred_y = tta_model.predict(test_x, batch_size=1, verbose=1)
    
    thre_list = np.load(os.path.join(PREPROCESSED, 'test5_notgenerator_densenet169_dense_TTA_thre.npy'))
    
    result_bool = ((test_pred_y - np.array(thre_list)) >= np.zeros((test_pred_y.shape)))
    
    submission_df = pd.read_csv(SAMPLE)
    
    for i in range(result_bool.shape[0]):
        ans_str = ""
        for j in range(result_bool.shape[1]):
            if result_bool[i,j]:
                ans_str += str(j)+" "
        submission_df.loc[i,'Predicted'] = ans_str[:-1]
        
    submission_df.to_csv(RESULT, index=False)