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
    TEST_X_PATH = sys.argv[1]    
    DICT_PATH = sys.argv[2]    
    OUTPUT_PATH = sys.argv[3]    
       
    w2v_model = 'w2v_ta_addtest.h5'
    emb_dim = 256
    MODELlist = ['ta_addtest_test2_LSTMGRU.h5', 'ta_addtest_test4_LSTMLSTM.h5', 'ta_addtest_test7_LSTMGRU.h5', 'ta_addtest_test8_LSTMGRU.h5', 'ta_addtest_test9_LSTMGRU.h5']
    
    ### Read files
    # test file
    with open(TEST_X_PATH, 'r', encoding='utf-8') as f:
        readin = f.readlines()
        # Use regular expression to get rid of the index
        test_sentences = [re.sub('^[0-9]+,', '', s) for s in readin[1:]]
        
    jieba.set_dictionary(DICT_PATH)  # Change dictionary (Optional)
    test_sentences = [list(jieba.cut(s, cut_all=False)) for s in test_sentences]
    
    emb_model = Word2Vec.load(w2v_model)
    
    # Convert words to index
    test_sentences_list = []
    for i, s in enumerate(test_sentences):
        toks = [emb_model.wv.vocab[w].index + 1 if w in emb_model.wv.vocab else 0 for w in s]  # Plus 1 to reserve index 0 for OOV words
        test_sentences_list.append(toks)
    
    # Pad sequence to same length
    max_length = 64
    test_sentences_array = pad_sequences(test_sentences_list, maxlen=max_length)
    
    X_test = test_sentences_array
    
    firsttime = True
    for m in MODELlist:
        model = load_model(m)
        Y_pred = model.predict(X_test, verbose=1)
        if firsttime:
            Y_pred_sum = Y_pred
            firsttime = False
        else:
            Y_pred_sum += Y_pred
        
    Y_pred = Y_pred_sum / len(MODELlist)
    pred_y_label = np.array([1 if y>0.5 else 0 for y in Y_pred])
    
    idlist = []
    for i in range(pred_y_label.shape[0]):
        idlist.append(i)
        
    df_submission = pd.DataFrame({'id':idlist,
                                  'label': pred_y_label  })
    
    df_submission.to_csv(OUTPUT_PATH,index=False)
