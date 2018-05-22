import numpy as np
import pandas as pd
from time import ctime
from os.path import join
from cPickle import load
import os,sys,gc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Input, Dense,Embedding,Dropout
from keras.layers.recurrent import GRU
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping

models = ['model1_cnn_seqlen_300_dicsize_20000',
          'model2_gru_seqlen_300_dicsize_20000',
          'model3_stack_seqlen_300_dicsize_20000']
n_models=len(models)

N_MODEL_PER_CLASS=10#used in model 3
print('\014')

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'
s = '_seqlen_300_dicsize_20000'

#===load data===
print(ctime()+'...loading data...')
with open(join(dataDir,'metadata'+s+'.pkl'),'rb') as f:
    dataDict = load(f)
DICT_SIZE,MAX_SEQ_LEN,class_names,n_train,n_test = (dataDict['DICT_SIZE'],
                                                    dataDict['MAX_SEQ_LEN'],
                                                    dataDict['class_names'],
                                                    dataDict['n_train'],
                                                    dataDict['n_test'])
X_train=np.fromfile(join(dataDir,'X_train'+s),'int32').\
                    reshape((n_train,MAX_SEQ_LEN))
X_test=np.fromfile(join(dataDir,'X_test'+s),'int32').\
                    reshape((n_test,MAX_SEQ_LEN))
Y_train=np.fromfile(join(dataDir,'Y'+s),'int8').reshape((n_train,6))                    
#===load data===

#===compute models predictions on train data===
pred_train = []

for i,mdl in enumerate(models):
    sys.stdout.write('model %d of %d' % (i,n_models))
    
    filepath=join(dataDir,'fittedModel_'+mdl)
    model = load_model(filepath) 
    if 'model3' in mdl:
        weakPred_train = np.fromfile(join(dataDir,
                                          'model3/models_pred_train')).\
                        reshape((X_train.shape[0],7*N_MODEL_PER_CLASS))
        pred_train.append(model.predict(weakPred_train,
                                        batch_size=100,verbose=1))
    else:
        pred_train.append(model.predict(X_train,batch_size=100,verbose=1))

    del model;gc.collect()

cols=[x+'_'+y for x in models for y in class_names]
pred_train = np.concatenate(pred_train,axis=1)
df=pd.DataFrame(pred_train);df.columns=cols
df.to_csv(join(dataDir,'models_pred_train.csv'),index=False)    
#===compute models predictions on train data===
    
#===train a model on top of other models' predictios===    
inputs = Input(shape=(n_models*6,))
out = Dense(6,activation='sigmoid')(inputs)
model = Model(inputs,out)
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])

early=EarlyStopping(monitor='val_loss', min_delta=0, patience=1)
filepath=join(dataDir,'fittedModel_stacked'+s)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             save_best_only=True, mode='min', period=1)
model.fit(pred_train,Y_train,batch_size=100,epochs=50,verbose=1,
          validation_split=.1,callbacks=[early,checkpoint])
#===train a model on top of other models' predictios=== 

#===compute predictions on test data===   
pred_test=[]
for mdl in models:
    df=pd.read_csv(join(dataDir,'submission_'+mdl+'.csv'))
    pred_test.append(df.iloc[:,1:].values)
pred_test=np.concatenate(pred_test,axis=1)

Y_test_pred=model.predict(pred_test,batch_size=100)
df_test=pd.read_csv(join(dataDir,'test.csv'))
df_test=pd.concat([df_test[['id']],pd.DataFrame(Y_test_pred)],axis=1)
df_test.columns=['id']+class_names

df_test.to_csv(join(dataDir,'submission_stacked.csv'),index=False)
#===compute predictions on test data===


