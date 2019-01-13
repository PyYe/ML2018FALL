wget -O ./src/ensemble/test_matches.csv https://www.dropbox.com/s/8jhjcl7lspzgq07/test_matches.csv?dl=1
wget -O ./src/model/test1_notgenerator.h5 https://www.dropbox.com/s/wapr9tsj0lyncyd/test1_notgenerator.h5?dl=1
wget -O ./src/model/test3_notgenerator_densenet201_dense.h5 https://www.dropbox.com/s/c7s0anepqnioom5/test3_notgenerator_densenet201_dense.h5?dl=1
wget -O ./src/model/test5_notgenerator_DenseNet169_dense.h5 https://www.dropbox.com/s/almv0z3xdekiyvq/test5_notgenerator_DenseNet169_dense.h5?dl=1
cp -r $1 ./src/data/train
cp -r $2 ./src/data/test
cp $3 ./src/data/train.csv
cp $4 ./src/data/sample_submission.csv