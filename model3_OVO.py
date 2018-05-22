'''
here for each class we train 10 models, where each model
is a binary classifier trained to classify that class
versus other classes. Each model is trained on a sample
which consists of equal number of positive and negative samples.
The 10 models are trained on 10 random samples obtained in this way.

We define a sample as `normal` if it is not in any of 6 classes.
For normal class, there are more positive samples than negative. For all
other classes there are more negatives. So for all calsses the random 
samples are obtained by keeping all positives and a random sample of 
the same size from negatives. For normal class we keep all negatives and 
a random sample of the positives.
'''

import numpy as np
from time import ctime
from os.path import join
from cPickle import load
import os,sys,gc

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Input, Dense,Embedding,Dropout
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping

print('\014')

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

#===add class 'normal'===
yn=np.zeros((Y_train.shape[0],1));yn[np.sum(Y_train,1)==0]=1;
Y_train=np.hstack((Y_train,yn))
class_names.append('normal')
#===add class 'normal'===

#===model===
def GRUnet(embed_dim,gru_units,dict_size,max_seq_len):    
    inputs = Input(shape=(max_seq_len,))
    embed = Embedding(input_dim=dict_size+1,output_dim=embed_dim,
                       input_length=max_seq_len)(inputs)
    rnn = GRU(units=gru_units)(embed)    
    out = Dense(1,activation='sigmoid')(rnn)
    model = Model(inputs=inputs,outputs=out)
    model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])
    return model

early=EarlyStopping(monitor='val_loss', min_delta=0, patience=1)
#===model===


#===build models for each class===
print(ctime()+'...training models...')
embed_dim=100;gru_units=10
des_idx=6
des_class=class_names[des_idx]
pos_idx = np.where(Y_train[:,des_idx]==1)[0]
neg_idx = np.where(Y_train[:,des_idx]                                                                          ==0)[0]
val_loss=[]
for i in range(N_MODEL_PER_CLASS):
    #===print progress===
    percent=i/float(N_MODEL_PER_CLASS-1)
    n1=int(percent*60);n2=60-n1;
    bar='% ['+'='*n1+'>'+'-'*n2+']'
    tabs='\t'*10;
    sys.stdout.write('\r %d %s \r' %(int(100*percent),bar))
    #===print progress===
    
    #===build random sample===
    if des_idx!=6:#if not normal
        n_idx=np.random.choice(neg_idx,size=pos_idx.size,replace=False)
        X_sampled = np.vstack((X_train[pos_idx,:],X_train[n_idx,:]))
        Y_sampled = np.concatenate((Y_train[pos_idx,des_idx],
                                    Y_train[n_idx,des_idx]))
    else:
        p_idx=np.random.choice(pos_idx,size=neg_idx.size,replace=False)
        X_sampled = np.vstack((X_train[p_idx,:],X_train[neg_idx,:]))
        Y_sampled = np.concatenate((Y_train[p_idx,des_idx],
                                    Y_train[neg_idx,des_idx]))
    X_sampled,Y_sampled=shuffle(X_sampled,Y_sampled)
    #===build random sample===
        
    #===train model===
    filepath=join(dataDir,'model3/cls_'+str(des_idx)+'_mdl_'+str(i))
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                 save_best_only=True, mode='min', period=1)
    model = GRUnet(embed_dim,gru_units,dict_size=DICT_SIZE,
                   max_seq_len=MAX_SEQ_LEN)
    train_history = model.fit(X_sampled,Y_sampled,batch_size=100,
                              epochs=10,verbose=1,
                              validation_split=.1,callbacks=[early,checkpoint])
    val_loss.append(min(train_history.history['val_acc']))
    del model;gc.collect()
    
print(np.mean(val_loss))
#===build models for each class===


    


