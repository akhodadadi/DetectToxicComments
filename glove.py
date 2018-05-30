from os.path import join
import numpy as np
import pandas as pd

from toxic.base import Toxic
from keras.preprocessing.text import Tokenizer
from time import ctime

embed_dim=50

dataDir='/home/arash/datasets/glove'
gloveDict = {}

fn=join(dataDir,'glove.6B.{}d.txt'.format(embed_dim))
glove_df=pd.read_csv(fn,engine='python',delim_whitespace=True,header=None)
glove_df.rename({0:'word'},axis=1,inplace=True)

        
tox = Toxic()
tox.loadData(textType='nw_excluded')

print(ctime()+'...tekenization...')
T=Tokenizer()
train_texts=tox.trainData.comment_text.tolist()
test_texts=tox.testData.comment_text.tolist()
T.fit_on_texts(train_texts+test_texts)

words_df=pd.DataFrame({'word':T.word_index.keys(),
                       'word_idx':T.word_index.values()})
df=pd.DataFrame({'word':T.word_counts.keys(),
                 'word_count':T.word_counts.values()})

words_df=words_df.merge(df,on='word')

words_df=words_df.merge(glove_df,on='word',how='left')
words_df.fillna(0,inplace=True)
words_df.sort_values(by='word_idx',inplace=True)
w=words_df.iloc[:,3:].values

        