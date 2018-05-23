#keras
from keras.layers import Input,Conv1D,MaxPool1D,Dense,Dropout,Embedding,\
Flatten,concatenate
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping

#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import numpy as np
from time import ctime
import pandas as pd
from . import toxic_config


#==================
#BASE CLASSES
#==================
class KerasModels():
    '''
    base class for keras models.
    '''
    def __init__(self,tox):
    
        self.model=None
        self.tox=tox    
        
    def fit(self,filepath,loss='binary_crossentropy',optimizer='adam',
            metrics=['accuracy'],batch_size=100,epochs=5,verbose=1,
            validation_split=.1,patience=1):
        
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             save_best_only=True, mode='min', period=1)
        early=EarlyStopping(monitor='val_loss',min_delta=0,patience=patience)
        
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        self.model.fit(x=self.tox.trainFeatureMat,y=self.tox.Y_train,
                       batch_size=batch_size,epochs=epochs,verbose=verbose,
                       validation_split=validation_split,
                       callbacks=[checkpoint,early])
    
    def predict(self,dstDir,batch_size=10000,loadModel=True,modelPath=None):
        
        if loadModel:
            self.model=load_model(modelPath)

        if len(self.tox.testData)==0:
            self.tox.loadData(loadTrain=False,loadTest=True)
            
        #===make predictions on test set===
        print(ctime()+'...computing prediction for test data...')
        Y_test=self.model.predict(self.tox.testFeatureMat,
                                  batch_size=batch_size,verbose=1)
        #===make predictions on test set===
            
        self.tox.testData=pd.concat([self.tox.testData[['id']],
                                     pd.DataFrame(Y_test)],axis=1)
        self.tox.testData.columns=['id']+toxic_config.CLASSES_NAME
        self.tox.testData.to_csv(dstDir,index=False)
        
    def __buildModel__(self,modelParams):
        pass
        

class decisionTreeModel():
    '''
    base class for decision tree models.
    '''
    
    def __init__(self,tox):
        self.model=None
        self.tox=tox 
        
    def fit(self,filepath,val_size=.1):
        
        #TODO: cross-validation
        print(ctime()+'...fitting model...')
        self.X_train,self.X_val,self.Y_train,self.Y_val=\
            train_test_split(self.tox.trainFeatureMat,
                             self.tox.Y_train,test_size=val_size)
        self.model.fit(self.X_train,self.Y_train)
        
        print(ctime()+'...saving fitted model to file...')
        joblib.dump(self.model,filepath+'.pkl')
        
    def predict(self,dstDir,loadModel=True,modelPath=None):
        
        if loadModel:
            self.model=joblib.load(modelPath)

        if len(self.tox.testData)==0:
            self.tox.loadData(loadTrain=False,loadTest=True)
            
        #===make predictions on test set===
        print(ctime()+'...computing prediction for test data...')
        Y_test = np.hstack([x[:,[1]] for x in \
                       self.model.predict_proba(self.tox.testFeatureMat)])
        #===make predictions on test set===
            
        self.tox.testData=pd.concat([self.tox.testData[['id']],
                                     pd.DataFrame(Y_test)],axis=1)
        self.tox.testData.columns=['id']+toxic_config.CLASSES_NAME
        self.tox.testData.to_csv(dstDir,index=False)        
        
        

#=======================
#keras models        
#=======================
class cnn_model(KerasModels):
    
    def __init__(self,tox,modelParams={'embed_dim':100,'n_featMap':500,
                                       'kernel_size':[3,4,5],
                                       'strides':[1]*3,'d_r':.1}):
    
        '''
        This function initialize a conv. neural netwok model with the 
        following architecture:
            The input is sequence of words. The first layer is an embedding
            layer. The next layer consists of 3 conv units each with 
            'n_featMap' feature maps and with different kernel size.
            Next is a max pooling layer, followed by a dropout layer
            and finally a dense layer which forms the output.             
            
        Parameters
        ----------  
        tox: base.Toxic
            An object of class base.Toxic.
        modelParams: dict
            it must contain the following fields:
                `embed_dim`: dimension of the output of emebdding layer,
                `n_featMap`: number of feature maps in the conv layer,
                `kernel_size`: a list of size 3, which specifies the kernel
                size of the three convolution units,
                `strides`: a list of size 3, which specifies the stride
                size of the three convolution units,
                'd_r': droupout rate               
        '''
        self.model=None
        self.tox=tox
        self.__buildModel__(modelParams)    
    
    def __buildModel__(self,modelParams):
        '''
        This function builds the model.
        '''
        
        #===model parameters===
        embed_dim,n_featMap,kernel_size,strides,d_r=\
            (modelParams['embed_dim'],
             modelParams['n_featMap'],
             modelParams['kernel_size'],
             modelParams['strides'],
             modelParams['d_r'])        
        #===model parameters===
        
        #===build model===
        #-input layer-
        max_seq_len=self.tox.trainFeatureMat.shape[1]
        dict_size=np.max((self.tox.trainFeatureMat.max(),
                          self.tox.testFeatureMat.max()))
        inputs = Input(shape=(max_seq_len,))
        #-input layer-
        
        #-embedding layer-
        embed = Embedding(input_dim=dict_size+1,output_dim=embed_dim,
                          input_length=max_seq_len)(inputs)
        #-embedding layer-
        
        #-convolutional units-
        conv0=Conv1D(filters=n_featMap,kernel_size=kernel_size[0],
                        strides=strides[0],activation='relu')(embed)
        conv1=Conv1D(filters=n_featMap,kernel_size=kernel_size[1],
                        strides=strides[1],activation='relu')(embed)
        conv2=Conv1D(filters=n_featMap,kernel_size=kernel_size[2],
                        strides=strides[2],activation='relu')(embed)
        #-convolutional units-
        
        #-max pool over all words-
        pool0 = MaxPool1D(pool_size=int(conv0.get_shape()[1]))(conv0)
        pool1 = MaxPool1D(pool_size=int(conv1.get_shape()[1]))(conv1)
        pool2 = MaxPool1D(pool_size=int(conv2.get_shape()[1]))(conv2)
        #-max pool over all words-
        
        #-dropuout and output-
        concat = concatenate([pool0,pool1,pool2])
        features = Dropout(d_r)(Flatten()(concat))
        out = Dense(6,activation='sigmoid')(features)
        #-dropuout and output-
        
        self.model = Model(inputs=inputs,outputs=out)
        #===build model===


#=======================
#decision tree models
#=======================  
class randomForestModel(decisionTreeModel):

    def __init__(self,tox,modelParams={'n_estimators':100,
                                       'min_samples_split':None,
                                       'class_weight':'balanced_subsample',
                                       'max_features':'auto',                                       
                                       'verbose':5}): 

        '''
        Constructor for a random forest model. 
        
        Parameters
        ----------  
        tox: base.Toxic
            An object of class base.Toxic.
        modelParams: dict
            parameters passed to `sklearn.ensemble.RandomForestClassifier`.
        '''       
        
        self.model=None
        self.tox=tox
        self.__buildModel__(modelParams)    
    
    def __buildModel__(self,modelParams):
        '''
        This function builds the model.
        '''
        
        #note that RandomForestClassifier supports multi-output 
        #classification.
        self.model=RandomForestClassifier(
                         n_estimators=modelParams['n_estimators'],
                         min_samples_split=modelParams['min_samples_split'],
                         class_weight=modelParams['class_weight'],
                         max_features=modelParams['max_features'],
                         verbose=modelParams['verbose'],
                         n_jobs=-1)
        
        
        
        