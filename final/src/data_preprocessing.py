import pandas as pd
import numpy as np
import os, sys, math, pickle
from PIL import Image
import random as rn

import tensorflow as tf
from tensorflow.image import resize_images



class data_generator:
    
    def create_train(dataset_info, train_size, shape, mode):
        
        random_indexes = np.random.choice(len(dataset_info), train_size)
        batch_images = np.empty((train_size, shape[0], shape[1], shape[2]), dtype=np.uint8)
        batch_labels = np.zeros((train_size, 28))
        for i, idx in enumerate(random_indexes):
            if mode=='RGBY':
                image = data_generator.RGBY(
                    dataset_info[idx]['path'], shape)   
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            elif mode=='RGB':
                image = data_generator.RGB(
                    dataset_info[idx]['path'], shape)   
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            else:
                print ("Unexpected mode!")

        return batch_images, batch_labels.astype(np.uint8)
            
    def create_test(dataset_info, test_size, shape, mode):
        batch_images = np.empty((test_size, shape[0], shape[1], shape[2]), dtype=np.uint8)
        batch_names = []
        for i, idx in enumerate(dataset_info):
            if mode=='RGBY':
                image = data_generator.RGBY(
                    idx['path'], shape)   
                batch_images[i] = image
                batch_names.append(idx['name'])
            elif mode=='RGB':
                image = data_generator.RGB(
                    idx['path'], shape)   
                batch_images[i] = image
                batch_names.append(idx['name'])
            else:
                print ("Unexpected mode!")

        return batch_images, batch_names
        
    def RGBY(path, shape):
        image_red_ch = np.asarray(Image.open(path+'_red.png'))
        image_green_ch = np.asarray(Image.open(path+'_green.png'))
        image_blue_ch = np.asarray(Image.open(path+'_blue.png'))
        image_yellow_ch = np.asarray(Image.open(path+'_yellow.png'))
        
        image = np.stack((
            image_red_ch, 
            image_green_ch, 
            image_blue_ch,
            image_yellow_ch
        ), -1)
        return image.astype(np.uint8)
    
    def RGB(path, shape):
        image_red_ch = np.asarray(Image.open(path+'_red.png'))
        image_green_ch = np.asarray(Image.open(path+'_green.png'))
        image_blue_ch = np.asarray(Image.open(path+'_blue.png'))
        
        image = np.stack((
            image_red_ch, 
            image_green_ch, 
            image_blue_ch
        ), -1)
        return image.astype(np.uint8)
                

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
    NAME = "data_preprocessing"
    PATH = os.getcwd()
    
    TRAIN = sys.argv[1]    
    TEST = sys.argv[2]    
    LABELS = sys.argv[3]    
    PREPROCESSED = sys.argv[4]    
    SAMPLE = sys.argv[5]    
    MODEL = sys.argv[6]    
    RESULT = sys.argv[7]  
    
    LABEL_NUM = 28
    IMAGE_LENGTH = 512
    IMAGE_WIDTH = 512
    CHANNEL_NUM = 4
    TRAIN_SIZE = int(len(os.listdir(TRAIN))/4)
    TEST_SIZE = int(len(os.listdir(TEST))/4)
    ### Read files
    # train files
    data = pd.read_csv(LABELS)

    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path':os.path.join(TRAIN, name),
            'labels':np.array([int(label) for label in labels])})
    train_dataset_info = np.array(train_dataset_info)
    
    test_dataset_info = []
    test_dataset_namelist = [ test_name.split('_')[0] for test_name in os.listdir(TEST)]
    test_name_unique = []
    
    prename = ""
    for name in test_dataset_namelist:
        if prename != name:
            test_name_unique.append(name)
            test_dataset_info.append({
                'path': os.path.join(TEST, name),
                'name': name })
            prename = name
    test_dataset_info = np.array(test_dataset_info)
    
    pickle.dump(test_name_unique, open(os.path.join(PREPROCESSED, 'test_name.pickle'), 'wb'))
    
    
    # create train datagen
    train_x, train_y= data_generator.create_train(
        dataset_info=train_dataset_info, train_size=TRAIN_SIZE, shape=(IMAGE_LENGTH,IMAGE_WIDTH,CHANNEL_NUM), mode='RGBY' )
    

    np.save(os.path.join(PREPROCESSED, 'train_RGBY_original_x.npy'), train_x)
    np.save(os.path.join(PREPROCESSED, 'train_RGBY_original_y.npy'), train_y)
    
    # create test datagen
    
    
    test_x, test_name= data_generator.create_test(
        dataset_info=test_dataset_info, test_size=TEST_SIZE, shape=(IMAGE_LENGTH,IMAGE_WIDTH,CHANNEL_NUM), mode='RGBY' )
    np.save(os.path.join(PREPROCESSED, 'test_RGBY_original_x.npy'), test_x)
    
    