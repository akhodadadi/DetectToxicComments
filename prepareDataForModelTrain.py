import pandas as pd
import numpy as np
from time import ctime
from os.path import join
from cPickle import dump

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

print('\014')

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'

#====configuration===
DICT_SIZE=20000
MAX_SEQ_LEN=300
#====configuration===

#===load data===
print(ctime()+'...loading data into pandas df...')
df_train=pd.read_csv(join(dataDir,'train_sw_excluded.csv'))
df_test=pd.read_csv(join(dataDir,'test_sw_excluded.csv'))
df_train.fillna('none',inplace=True)
df_test.fillna('none',inplace=True)
class_names=df_train.columns[2:].tolist()
#===load data===

#===tokenize and generate sequences===
texts_train = df_train.comment_text.tolist()
texts_test = df_test.comment_text.tolist()
Y = np.array(df_train.iloc[:,2:],dtype='int8')

T = Tokenizer(num_words=DICT_SIZE)
print(ctime()+'...tekenization...')
T.fit_on_texts(texts_train+texts_test)

print(ctime()+'...convert texts to sequeces...')
sequences_train = T.texts_to_sequences(texts_train)
sequences_test = T.texts_to_sequences(texts_test)

X = pad_sequences(sequences=sequences_train,
                        maxlen=MAX_SEQ_LEN,padding='pre')
X_test = pad_sequences(sequences=sequences_test,
                        maxlen=MAX_SEQ_LEN,padding='pre')
#===tokenize and generate sequences===

#===save data===
print(ctime()+'...save data to file...')
s='_seqlen_'+str(MAX_SEQ_LEN)+'_dicsize_'+str(DICT_SIZE)

X.tofile(join(dataDir,'X_train'+s))
X_test.tofile(join(dataDir,'X_test'+s))
Y.tofile(join(dataDir,'Y'+s))

metadata={'class_names':class_names,'DICT_SIZE':DICT_SIZE,
          'MAX_SEQ_LEN':MAX_SEQ_LEN,'n_train':X.shape[0],
          'n_test':X_test.shape[0]}
with open(join(dataDir,'metadata'+s+'.pkl'),'wb') as f:
    dump(metadata,f)
print(ctime()+\
      '...DONE! data is saved in:\n %s'%dataDir)
#===save data===

