'''
Here we build a model which takes the predictions of the 
one-versus-others classifiers as input, and makes the 
final prediction.
'''

import pandas as pd
import numpy as np
from time import ctime
from os.path import join
from cPickle import load
import os,sys,gc

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputClassifier

import xgboost as xgb

from keras.layers import Input, Dense,Embedding,Dropout
from keras.layers.recurrent import GRU
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping

print('\014')

ifLoadPredFromFile=True

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'

s = '_seqlen_300_dicsize_20000'

if not os.path.exists(join(dataDir,'model3')):
   os.mkdir(join(dataDir,'model3')) 

#====configuration===
N_MODEL_PER_CLASS=10#no. of models trained per class
#====configuration===

#===load data===
print(ctime()+'...loading data...')
with open(join(dataDir,'metadata'+s+'.pkl'),'rb') as f:
    dataDict = load(f)
DICT_SIZE,MAX_SEQ_LEN,class_names,n_train,n_test = (dataDict['DICT_SIZE'],
                                                    dataDict['MAX_SEQ_LEN'],
                                                    dataDict['class_names'],
                                                    dataDict['n_train'],
                                                    dataDict['n_test'])
X=np.fromfile(join(dataDir,'X_train'+s),'int32').\
                    reshape((n_train,MAX_SEQ_LEN))
X_test=np.fromfile(join(dataDir,'X_test'+s),'int32').\
                    reshape((n_test,MAX_SEQ_LEN))
Y=np.fromfile(join(dataDir,'Y'+s),'int8').reshape((n_train,6))                    
#===load data===

#===split data into train and validation===
#X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=.1,shuffle=True)
X_train=X;Y_train=Y
#===split data into train and validation===

#===compute predictions or load from file===
if ifLoadPredFromFile:
    pred_train = np.fromfile(join(dataDir,'model3/models_pred_train')).\
                    reshape((X_train.shape[0],7*N_MODEL_PER_CLASS))
    pred_test = np.fromfile(join(dataDir,'model3/models_pred_test')).\
                    reshape((X_test.shape[0],7*N_MODEL_PER_CLASS))                    
else:
    pred_train = np.zeros((X_train.shape[0],7*N_MODEL_PER_CLASS))
    pred_test = np.zeros((X_test.shape[0],7*N_MODEL_PER_CLASS))
    k=0
    for des_idx in range(7):
        for i in range(N_MODEL_PER_CLASS):
            #===print progress===
            percent1=i/float(N_MODEL_PER_CLASS-1)
            n1=int(percent1*60);n2=60-n1;
            bar1='% ['+'='*n1+'>'+'-'*n2+']'
            
            percent2=des_idx/7.0
            n1=int(percent2*60);n2=60-n1;
            bar2='% ['+'='*n1+'>'+'-'*n2+']'        
            tabs='\t'*10;
            sys.stdout.write('\r class %d: %d %s %s total: %d %s \r' %\
                             (des_idx,int(100*percent1),bar1,tabs,
                              int(100*percent2),bar2))
            #===print progress===
            
            #===compute predictions for train and test data===
            filepath=join(dataDir,'model3/cls_'+str(des_idx)+'_mdl_'+str(i))
            model = load_model(filepath)        
            y_pred=model.predict(X_train,batch_size=100)
            pred_train[:,[k]] = y_pred
            y_pred=model.predict(X_test,batch_size=100)
            pred_test[:,[k]] = y_pred
            del model,y_pred;gc.collect()
            k+=1
            #===compute predictions for train and test data===
        
    pred_train.tofile(join(dataDir,'model3/models_pred_train'))
    pred_test.tofile(join(dataDir,'model3/models_pred_test'))
#===compute predictions or load from file===
    
#===build model===
inputs = Input(shape=(7*N_MODEL_PER_CLASS,))
out = Dense(6,activation='sigmoid')(inputs)
model = Model(inputs,out)
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])

early=EarlyStopping(monitor='val_loss', min_delta=0, patience=1)
filepath=join(dataDir,'fittedModel_model3_stack'+s)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             save_best_only=True, mode='min', period=1)
model.fit(pred_train,Y,batch_size=100,epochs=50,verbose=1,
          validation_split=.1,callbacks=[early,checkpoint])
print(ctime()+'...computing prediction for test data...')
Y_test_pred=model.predict(pred_test,batch_size=100)    
#===build model===
    
    
##===build xgboost model===
#print(ctime()+'...fitting xgboost...')
#xgb_clf=xgb.XGBClassifier(subsample=.5,colsample_bytree=.6z,silent=False)
#clf = MultiOutputClassifier(xgb_clf,n_jobs=-1)
#clf.fit(pred_train,Y_train)
#acc=clf.score(pred_train,Y_train)
#print('mean acc on train is: %.3f' % acc) 
#print(ctime()+'...computing prediction for test data...')
#Y_test_pred=clf.predict(pred_test)
##===build xgboost model===

#===compute predictions for test data===
df_test=pd.read_csv(join(dataDir,'test.csv'))
df_test=pd.concat([df_test[['id']],pd.DataFrame(Y_test_pred)],axis=1)
df_test.columns=['id']+class_names

df_test.to_csv(join(dataDir,'submission_model3_stack'+s+'.csv'),index=False)
#===compute predictions for test data===
    
    






    

