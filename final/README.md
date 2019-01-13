# 執行環境
windows 10 anaconda3 jupyter notebook python3.6.6

# Download files on dropbox
bash download.sh [data/train] [data/test] [data/train.csv] [data/samplesubmission.csv]

# Folders information
./src/data                     將kaggle上的data zips 解壓縮並存在這個資料夾  
./src/ensemble                 執行ensemble處  
./src/model                    存train好的model  
./src/preprocessed_data        存前處理過的data  
./src/result                   存結果    
		
# externel tools:

* scikit-learn 
    * 0.20.1
	
* pillow 
    * 5.3.0

* OpenCV
    * 3.4.3

* tta_wrapper
    * $pip install tta-wrapper

# 執行順序

* data_preprocessing.ipynb
* data_analysis.ipynb

* 為train的檔案 可不執行 直接用dropbox下載下來的model  
** test1_notgenerator.ipynb

* test1_notgenerator_TTA_thre.ipynb
* test1_notgenerator_TTA_test.ipynb

* 為train的檔案 可不執行 直接用dropbox下載下來的model  
** test3_notgenerator_densenet201_dense.ipynb

* test3_notgenerator_densenet201_dense_TTA_thre.ipynb
* test3_notgenerator_densenet201_dense_TTA_test.ipynb

* 為train的檔案 可不執行 直接用dropbox下載下來的model  
** test5_notgenerator_DenseNet169_dense.ipynb

* test5_notgenerator_densenet169_dense_TTA_thre.ipynb
* test5_notgenerator_densenet169_dense_TTA_test.ipynb

# Ensemble
bash pre_ensemble.sh (將./src/result中的檔案 複製到 ./src/ensemble/data裡)
執行 ./src/ensemble/ensemble.ipynb

# 最後上傳kaggle檔案
* 位於./src/ensemble 的 extra_submission_df.csv
