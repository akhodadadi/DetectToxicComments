from time import ctime
import pandas as pd
from os.path import join
from scipy.sparse import load_npz
import numpy as np
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from  xgboost import XGBClassifier
import lightgbm as lgb

print('\014') 

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'

#===load data===
print(ctime()+'...loading data...')
X_train = load_npz(join(dataDir,'train_tfidf.npz'))
X_test = load_npz(join(dataDir,'test_tfidf.npz'))

df_train=pd.read_csv(join(dataDir,'train.csv'))
class_names = df_train.columns[2:].tolist()
Y_train = np.array(df_train.iloc[:,2:],dtype='int8')
del df_train;gc.collect()
#===load data===

#===split data into train and validation===
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=.1)
#===split data into train and validation===

#===train model===
print(ctime()+'...fitting random forest...')
#clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100,
#                                                   max_depth=None,verbose=1,
#                                                   n_jobs=-1))

#xgb_clf=XGBClassifier(n_estimators=100,eta=.1,subsample=.5)
#clf = MultiOutputClassifier(xgb_clf)

clf=lgb.LGBMClassifier(bagging_fraction=0.8,feature_fraction=.8,
                       learning_rate=.1,num_leaves=31,
                       min_split_gain=.1,reg_alpha=.1)
clf = MultiOutputClassifier(clf)

clf.fit(X_train,Y_train)
print(ctime()+'...DONE!')
Y_val_pred = np.hstack([x[:,[1]] for x in clf.predict_proba(X_val)])
print('validation roc_auc_score=%.3f'%roc_auc_score(Y_val,Y_val_pred))
#===train model===

#===compute predictions for test data===
print(ctime()+'...computing prediction for test data...')
Y_test_pred = np.hstack([x[:,[1]] for x in clf.predict_proba(X_test)])
df_test=pd.read_csv(join(dataDir,'test.csv'))
df_test=pd.concat([df_test[['id']],pd.DataFrame(Y_test_pred)],axis=1)
df_test.columns=['id']+class_names

df_test.to_csv(join(dataDir,'submission_model6_RF.csv'),index=False)
#===compute predictions for test data===
