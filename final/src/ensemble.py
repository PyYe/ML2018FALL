import pandas as pd
import numpy as np
import os 


if __name__ == "__main__":
    PATH = os.getcwd()
    DATA = sys.argv[1]        
    anslist = os.listdir(DATA)
    
    thre = 2
    test_no = 11702
    class_num = 28
    
    vote_array = np.zeros((test_no, class_num))
    
    for ans in anslist:
        df = pd.read_csv(os.path.join(DATA, ans))
        for i in range(test_no):
            try:
                #print (df.loc[i, 'Predicted'])
                selectlist = df.loc[i, 'Predicted'].split(' ')
                for s in selectlist:
                    vote_array[i, int(s)] += 1
            except:
                continue
            
    submission_df = pd.read_csv(os.path.join(DATA, anslist[1]))

    for i in range(vote_array.shape[0]):
        ans_str = ""
        for j in range(vote_array.shape[1]):
            if vote_array[i,j] >= thre:
                ans_str += str(j)+" "
        submission_df.loc[i,'Predicted'] = ans_str[:-1]
        
    extra_df = pd.read_csv('test_matches.csv')
    
    #submission_df.to_csv('submission_df.csv', index=False)
    
    extra_submission_df = pd.DataFrame.copy(submission_df)
    
    for test_id, target in zip(extra_df['Test'], extra_df['Target']):
        extra_submission_df.loc[submission_df['Id']==test_id, 'Predicted'] = target
        
    extra_submission_df.to_csv('extra_submission_df.csv', index=False)