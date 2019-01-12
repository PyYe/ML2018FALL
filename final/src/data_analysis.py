import os
import numpy as np
import pandas as pd



def load_data(data_path):
    class_dict = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        readin = f.readlines()
        # Use regular expression to get rid of the index
        id_list = [s.split(',')[0] for s in readin[1:]]
        for s in readin[1:]:
            classes = s.split(',')[1]
            classes_list = classes.strip('\n').split(' ')
            for c in classes_list:
                if class_dict.get(c):
                    class_dict[c] += 1
                else:
                    class_dict[c] = 1
    return id_list, class_dict

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
    NAME = "data_analysis"
    PATH = os.getcwd()
    
    TRAIN = sys.argv[1]    
    TEST = sys.argv[2]    
    PREPROCESSED = sys.argv[3]    
    LABELS = sys.argv[4]    
    SAMPLE = sys.argv[5]    
    CLASS_WEIGHT = sys.argv[6]    
    
    LABEL_NUM = 28
    ### Read files
    # train files
    id_list, class_dict = load_data(LABELS)
    
    class_list = []
    for key in class_dict:
        for i in range(class_dict[key]):
            class_list.append(int(key))
        
    from sklearn.utils.class_weight import compute_class_weight

    class_weight = compute_class_weight('balanced', range(28), np.array(class_list))
    
    np.save(CLASS_WEIGHT, class_weight)
    
    train_x = np.load(os.path.join(PREPROCESSED, 'train_RGBY_original_x.npy'))
    train_y = np.load(os.path.join(PREPROCESSED, 'train_RGBY_original_y.npy'))
    
    valid_x = train_x[int(train_x.shape[0]*0.9):,:]
    valid_y = train_y[int(train_y.shape[0]*0.9):,:]
    
    np.save(os.path.join(PREPROCESSED, 'valid_RGBY_original_x.npy'), valid_x)
    np.save(os.path.join(PREPROCESSED, 'valid_RGBY_original_y.npy'), valid_y)
    