#!/bin/bash
wget -O w2v_ta_addtest.h5 https://www.dropbox.com/s/0cbco8ao6k7rqne/w2v_ta_addtest.h5?dl=1
wget -O ta_addtest_test7_LSTMGRU.h5 https://www.dropbox.com/s/5qqlwdgne05iufn/ta_addtest_test7_LSTMGRU.h5?dl=1
python3 hw4_test.py $1 $2 $3