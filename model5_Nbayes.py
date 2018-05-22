from time import ctime
import pandas as pd
from os.path import join
from scipy.sparse import load_npz
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split

print('\014') 

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'

#===load data===
X_train = load_npz(join(dataDir,'train_tfidf.npz'))
X_test = load_npz(join(dataDir,'test_tfidf.npz'))

df_train=pd.read_csv(join(dataDir,'train.csv'))
Y_train = np.array(df_train.iloc[:,2:],dtype='int8')
del df_train
#===load data===

#===split data into train and validation===
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=.1)
#===split data into train and validation===

#===train model===
print(ctime()+'...fitting MultinomialNB...')
clf = MultiOutputClassifier(MultinomialNB(alpha=.001))
clf.fit(X_train,Y_train)
print(ctime()+'...DONE!')
Y_val_pred = np.hstack([x[:,[1]] for x in clf.predict_proba(X_val)])
print('validation roc_auc_score=%.3f'%roc_auc_score(Y_val,Y_val_pred))
#===train model===

