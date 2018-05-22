'''
This scrtip computes bag of word representation of train and test
data.
'''
from time import ctime
import pandas as pd
from os.path import join
from scipy.sparse import save_npz,hstack
import gc


from sklearn.feature_extraction.text import TfidfVectorizer
print('\014')   

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'

#===load data===
print(ctime()+'...loading data into pandas df...')
df_train=pd.read_csv(join(dataDir,'train.csv'))
df_test=pd.read_csv(join(dataDir,'test.csv'))

train_text = df_train.comment_text.tolist()
test_text = df_test.comment_text.tolist()
all_texts=train_text+test_text
#===load data===

#===compute tfidf===
print(ctime() + '...computing words tf-idf...')
word_vectorizer = TfidfVectorizer(
                                    sublinear_tf=True,
                                    strip_accents='unicode',
                                    analyzer='word',
                                    token_pattern=r'\w{1,}',
                                    stop_words='english',
                                    ngram_range=(1, 1),
                                    max_features=20000)
word_vectorizer.fit(all_texts)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print(ctime() + '...computing chars tf-idf...')
char_vectorizer = TfidfVectorizer(
                                    sublinear_tf=True,
                                    strip_accents='unicode',
                                    analyzer='char',
                                    stop_words='english',
                                    ngram_range=(2, 4),
                                    max_features=20000)
char_vectorizer.fit(all_texts)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features=hstack([train_char_features,train_word_features])
test_features=hstack([test_char_features,test_word_features])
del train_char_features,test_char_features,train_word_features,test_word_features
gc.collect()
print(ctime() + '...DONE!')
#===compute tfidf===

#===save to file===
print(ctime() + '...saving to file...')
save_npz(join(dataDir,'train_tfidf.npz'), train_features)
save_npz(join(dataDir,'test_tfidf.npz'), test_features)
print(ctime() + '...DONE!')
del test_features,train_features
gc.collect()
#===save to file===






