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

def calcu_thre(y, pred_y):
    col_n = y.shape[1]
    thre_list = []
    for i in range(col_n):
        print (i, end=" ")
        y_list = list(y[:,i])
        pred_list = list(pred_y[:,i])
        pred_sorted_list = sorted(pred_list)
        median_list = [min(pred_sorted_list)-0.005]
        
        # calculate probably threshold
        for p in range(len(pred_sorted_list)-1):
            median_list.append( (pred_sorted_list[p]+pred_sorted_list[p+1])/2 )
        median_list.append(max(pred_sorted_list)+0.005)
        
        # find the threshold to maximun accuracy
        best_acc = 0
        for m in median_list:
            
            tf_list = list(np.array(y_list) == np.array([ 1 if pre>=m else 0 for pre in pred_list]))
            tf_list = [1 if t else 0 for t in tf_list]
            acc = sum(tf_list)/len(tf_list)
            if acc > best_acc:
                best_acc = acc
                best_thre = m
                    
        thre_list.append(best_thre)
    return thre_list

def f1_np(y_pred, y_true, threshold=0.5):
    '''numpy f1 metric'''
    y_pred = (y_pred>threshold).astype(int)
    TP = (y_pred*y_true).sum(1)
    prec = TP/(y_pred.sum(1)+1e-7)
    rec = TP/(y_true.sum(1)+1e-7)
    res = 2*prec*rec/(prec+rec+1e-7)
    return res.mean()


def f1_n(y_pred, y_true, thresh, n, default=0.5):
    '''partial f1 function for index n'''
    threshold = default * np.ones(y_pred.shape[1])
    threshold[n]=thresh
    return f1_np(y_pred, y_true, threshold)

def sub_find(args):
    y_pred = args[0]
    y_true = args[1]
    th = args[2]
    i = args[3]
    aux = f1_n(y_pred, y_true, th, i)
    return aux

def find_thresh(y_pred, y_true):
    '''brute force thresh finder'''
    ths = []
    for i in range(y_pred.shape[1]):
        args = []
        for th in np.linspace(0, 1, 1000):
            args.append((y_pred, y_true, th, i))
        with Pool() as p:
            aux = p.map(sub_find, args)
        
#         aux = []
#         for th in np.linspace(0, 1, 1000):
#             aux += [f1_n(y_pred, y_true, th, i)]
        ths += [np.array(aux).argmax() / 1000]
    return np.array(ths)

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
    NAME = "test5_notgenerator_densenet169_dense_TTA_thre"
    PATH = os.getcwd()
       
    PREPROCESSED = sys.argv[1]        
    MODEL = sys.argv[2]    
    
    IMAGE_LENGTH = 512
    IMAGE_WIDTH = 512
    CHANNEL_NUM = 4
    LABEL_NUM = 28
    
    import os
    if os.name is 'nt':
        from multiprocessing.dummy import Pool
    else:
        from multiprocessing import Pool
    
    
    valid_x = np.load(os.path.join(PREPROCESSED, 'valid_RGBY_original_x.npy'))
    valid_y = np.load(os.path.join(PREPROCESSED, 'valid_RGBY_original_y.npy'))
    
    model = load_model(MODEL, custom_objects={'f1': f1, 'focal_loss_fixed' : focal_loss()})
    
    tta_model = tta_classification(model, h_flip=True, rotation=(90, 270), 
                             merge='mean')


    valid_pred_y = tta_model.predict(valid_x, batch_size=1, verbose=1)

    thre_list = find_thresh(valid_pred_y, valid_y)
    np.save(os.path.join(PREPROCESSED, NAME+'.npy'), thre_list)