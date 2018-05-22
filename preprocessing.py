'''
this script removes stop words and extra characters from texts
in the train and test set and save the results in .csv files.
'''

import pandas as pd
from os.path import join  
from time import ctime 
import gc 
from nltk.corpus import stopwords
import re
import toxic_config
sw = stopwords.words('english')
print('\014')   


removeStopWords=False
dataDir = toxic_config.DATADIR

def cleanText(text,removeStopWords=True):
    '''
    this function removes extra characters and stop-words.
    '''
    
    text = re.sub('[\W]',' ',text)
    if removeStopWords:
        tokens = filter(lambda x:(x.lower() not in sw) and (len(x)>2),
                        text.split())
        return ' '.join(tokens)
    else:
        return text


#===load data===
print(ctime()+'...loading data into pandas df...')
df_train=pd.read_csv(join(dataDir,'train.csv'))
df_test=pd.read_csv(join(dataDir,'test.csv'))
#===load data===

#===clean data===
if removeStopWords:
    file_ext='sw_excluded'#stop words and non-words excluded
else:
    file_ext='nw_excluded'#only non-words excluded
    
print(ctime()+'...cleaning train data...')
df_train['comment_text'] = df_train['comment_text'].apply(cleanText)
print(ctime()+'...saving train data...')
df_train.to_csv(join(dataDir,'train_{}.csv'.format(file_ext)),index=False)
del df_train;gc.collect()

print(ctime()+'...cleaning test data...')
df_test['comment_text'] = df_test['comment_text'].apply(cleanText)
print(ctime()+'...saving test data...')
df_test.to_csv(join(dataDir,'test_{}.csv'.format(file_ext)),index=False)
del df_test;gc.collect()
#===clean data===




