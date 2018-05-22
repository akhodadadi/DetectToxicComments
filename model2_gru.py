import pandas as pd
import numpy as np
from time import ctime
from os.path import join
from cPickle import load

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense,Embedding,Dropout
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping

print('\014')

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'

s = '_seqlen_300_dicsize_20000'

#====configuration===
d_r=.5#dropout rate
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

#===build model===
inputs = Input(shape=(MAX_SEQ_LEN,))
embed = Embedding(input_dim=DICT_SIZE+1,output_dim=100,
                  input_length=MAX_SEQ_LEN)(inputs)


rl1=GRU(units=100,return_sequences=True)(embed)
rl2=GRU(units=100)(rl1)

dense1 = Dropout(d_r)(Dense(100,activation='relu')(rl2))
dense2 = Dense(50,activation='relu')(dense1)

out = Dense(6,activation='sigmoid')(dense2)

model = Model(inputs=inputs,outputs=out)
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])
#===build model===

#===train model===
filepath=join(dataDir,'fittedModel_model2_gru'+s)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             save_best_only=True, mode='min', period=1)
early=EarlyStopping(monitor='val_loss', min_delta=0, patience=1)
model.fit(x=X_train,y=Y_train,batch_size=100,epochs=5,verbose=1,
          validation_split=.1,callbacks=[checkpoint,early])
#===train model===

#===compute ROC-AUC for train data===
#print(ctime()+'...computing prediction for validation data...')
#Y_val_pred=model.predict(X_val)
#roc_auc_val=roc_auc_score(Y_val,Y_val_pred,average=None)
#print('AUC for validation set is %.4f'%np.mean(roc_auc_val))

print(ctime()+'...computing prediction for test data...')
Y_test_pred=model.predict(X_test,batch_size=100)
df_test=pd.read_csv(join(dataDir,'test.csv'))
df_test=pd.concat([df_test[['id']],pd.DataFrame(Y_test_pred)],axis=1)
df_test.columns=['id']+class_names

df_test.to_csv(join(dataDir,'submission_model2_gru'+s+'.csv'),index=False)
#===compute ROC-AUC for train data===













