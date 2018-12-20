import os, sys, re
import random as rn
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.layers import LSTM, CuDNNLSTM, Bidirectional, CuDNNGRU, Convolution1D, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.backend import hard_sigmoid
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

import jieba
from gensim.models import Word2Vec

def load_data(data_path, label_path):

    with open(data_path, 'r', encoding='utf-8') as f:
        readin = f.readlines()
        # Use regular expression to get rid of the index
        sentences = [re.sub('^[0-9]+,', '', s) for s in readin[1:]]
        
    labels = pd.read_csv(label_path)['label']
    labels = np.array(labels)
    
    return sentences, labels

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
    TRAIN_X_PATH = sys.argv[1]    
    TRAIN_Y_PATH = sys.argv[2]    
    TEST_X_PATH = sys.argv[3]    
    DICT_PATH = sys.argv[4]    
    
    w2v_model = 'w2v_ta_addtest.h5'
    emb_dim = 256
    MODEL = 'ta_addtest_test7_LSTMGRU.h5'
    ### Read files
    # train files
    print ('Loading train files ...')
    train_sentences, train_labels = load_data(TRAIN_X_PATH, TRAIN_Y_PATH)    
    
    # test file
    print ('Loading test file ...')
    with open(TEST_X_PATH, 'r', encoding='utf-8') as f:
        readin = f.readlines()
        # Use regular expression to get rid of the index
        test_sentences = [re.sub('^[0-9]+,', '', s) for s in readin[1:]]
        
    sentences = train_sentences + test_sentences
    
    
    jieba.set_dictionary(DICT_PATH)  # Change dictionary (Optional)
    print ('Jieba cutting all sets ...')
    sentences = [list(jieba.cut(s, cut_all=False)) for s in sentences]

    # Train Word2Vec model
    print ('Training Word2Vec model ...')
    emb_model = Word2Vec(sentences, size=emb_dim)
    emb_model.save(w2v_model)
    
    print ('Jieba cutting train set ...')
    train_sentences = [list(jieba.cut(s, cut_all=False)) for s in train_sentences]
    
    num_words = len(emb_model.wv.vocab) + 1  # +1 for OOV words
    emb_dim = emb_model.vector_size
    # Create embedding matrix (For Keras)
    print ('Creating embedding matrix (For Keras) ...')
    emb_matrix = np.zeros((num_words, emb_dim), dtype=float)
    for i in range(num_words - 1):
        v = emb_model.wv[emb_model.wv.index2word[i]]
        emb_matrix[i+1] = v   # Plus 1 to reserve index 0 for OOV words
    
    # Convert words to index
    train_sequences = []
    for i, s in enumerate(train_sentences):
        toks = [emb_model.wv.vocab[w].index + 1 if w in emb_model.wv.vocab else 0 for w in s]  # Plus 1 to reserve index 0 for OOV words
        train_sequences.append(toks)
    
    # Pad sequence to same length
    max_length = 64
    train_sequences = pad_sequences(train_sequences, maxlen=max_length)
    
    # Split validation data (#TODO)
    X_train = train_sequences[:int(len(train_labels)*0.9)]
    X_val = train_sequences[int(len(train_labels)*0.9):]
    Y_train = train_labels[:int(len(train_labels)*0.9)]
    Y_val = train_labels[int(len(train_labels)*0.9):]
    
    
    model = Sequential()
    model.add(Embedding(num_words,
                        emb_dim,
                        weights=[emb_matrix],
                        input_length=max_length,
                        trainable=False))
                        
    #########################################################
    # Design your own model!
    #model.add(CuDNNLSTM(256, return_sequences=True))#, dropout=0.5, recurrent_dropout=0.5))
    #model.add(Convolution1D(256, 3, border_mode='same'))
    model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(GRU(256, dropout=0.5, recurrent_dropout=0.5))
    
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
   
    #########################################################
    
    # Setting optimizer and compile the model
    adam = Adam(lr=0.0001, decay=1e-6, clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # Setting callback functions
    epochs = 2
    batch_size = 32
    #csv_logger = CSVLogger(LOGGER)
    checkpoint = ModelCheckpoint(filepath=MODEL,
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 mode='max')
    earlystopping = EarlyStopping(monitor='val_acc', 
                                  patience=6, 
                                  verbose=1, 
                                  mode='max')
                                 
    # Pre_train the model without embedding layers
    print ('Fitting first time...')
    fitHistory = model.fit(X_train, Y_train, 
              validation_data=(X_val, Y_val),
              epochs=epochs, 
              batch_size=batch_size,
              callbacks=[earlystopping, checkpoint]
             )#, csv_logger])
    
    for layer in model.layers:
        layer.trainable = True
        
    adam = Adam(lr=0.0001, decay=1e-6, clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
        
    ### training
    epochs = 1000
    batch_size = 32
    #csv_logger = CSVLogger(LOGGER)
    checkpoint = ModelCheckpoint(filepath=MODEL,
                                 verbose=1,
                                 save_best_only=True,
                                 monitor='val_acc',
                                 mode='max')
    earlystopping = EarlyStopping(monitor='val_acc', 
                                  patience=6, 
                                  verbose=1, 
                                  mode='max')
                                 
    # Train the model
    print ('Fitting last time...')
    fitHistory = model.fit(X_train, Y_train, 
              validation_data=(X_val, Y_val),
              epochs=epochs, 
              batch_size=batch_size,
              callbacks=[earlystopping, checkpoint]
             )#, csv_logger])
    
    
