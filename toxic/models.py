#keras
from keras.layers import Input,Conv1D,MaxPool1D,Dense,Dropout,Embedding,\
Flatten,concatenate,GRU
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import regularizers

#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputClassifier

from lightgbm import LGBMClassifier

import numpy as np
from time import ctime
import pandas as pd
from os.path import join
from . import toxic_config


def ensembleModels(modelsOutDir,filenames,weights):
    '''
    This function computes the weighed average of the output of 
    several other outputs.
    
    Parameters
    --------
    modelsOutDir: str
        Directory where the output of the other models are saved.
    filenames: list
        List of the filenames of the output of other models.
    weights: array like
        weights of the models.
    '''
    
    for i,fn in enumerate(filenames):
        if i==0:
            df_en = pd.read_csv(join(modelsOutDir,fn))
            df_en.iloc[:,1:]=weights[i]*df_en.iloc[:,1:]
        else:
            df = pd.read_csv(join(modelsOutDir,fn))
            df_en.iloc[:,1:]=df_en.iloc[:,1:]+weights[i]*df.iloc[:,1:]

    df_en.iloc[:,1:]=df_en.iloc[:,1:]/np.sum(weights)
    df_en.to_csv(join(modelsOutDir,'ensemble.csv'),index=False)
    
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
    
    def __init__(self,tox,modelParams):
        self.model=None
        self.tox=tox
        self.__buildModel__(modelParams) 
        
    def fit(self,filepath,val_size=.1,monitor_eval=False,fitParams={}):
        
        #TODO: cross-validation
        print(ctime()+'...fitting model...')
        self.X_train,self.X_val,self.Y_train,self.Y_val=\
            train_test_split(self.tox.trainFeatureMat,
                             self.tox.Y_train,test_size=val_size)
        if monitor_eval:
            fitParams.update({'eval_set':[(self.X_val,self.Y_val)]})
            self.model.fit(self.X_train,self.Y_train,sample_weight=None,
                           estimatorParam=fitParams)    
        else:
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
                                       'strides':[1]*3,'d_r':.1,
                                       'l2_reg':1.}):
    
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
        embed_dim,n_featMap,kernel_size,strides,d_r,l2_reg=\
            (modelParams['embed_dim'],
             modelParams['n_featMap'],
             modelParams['kernel_size'],
             modelParams['strides'],
             modelParams['d_r'],
             modelParams['l2_reg'])
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
        out = Dense(6,activation='sigmoid',
                    kernel_regularizer=regularizers.l2(l2_reg))(features)
        #-dropuout and output-
        
        self.model = Model(inputs=inputs,outputs=out)
        #===build model===

class gru_model(KerasModels):
    
    def __init__(self,tox,modelParams={'embed_dim':100,'n_dense':50,
                                       'n_units1':100,'n_units2':100}):
    
        '''
        This function initialize a neural netwok model with the 
        following architecture:
            The input is sequence of words. The first layer is an embedding
            layer. The input layer is followed by two layers od GRU units.
            The output of the last one is fed to a fully connected layer
            with relu activation and ginally an output layer which
            is a fully connected layer with sigmoid activation functions.
             
            
        Parameters
        ----------  
        tox: base.Toxic
            An object of class base.Toxic.
        modelParams: dict
            it must contain the following fields:
                `embed_dim`: dimension of the output of emebdding layer,
                `n_units1`: number of neurons in the first GRU layer.
                `n_units2`: number of neurons in the second GRU layer.
                'n_dense': number of neurons of the dense layer
        '''
        self.model=None
        self.tox=tox
        self.__buildModel__(modelParams)  
        
    def __buildModel__(self,modelParams):
        '''
        This function builds the model.
        '''
        
        #===model parameters===
        embed_dim,n_units1,n_units2,n_dense=\
            (modelParams['embed_dim'],
             modelParams['n_units1'],
             modelParams['n_units2'],
             modelParams['n_dense'])
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

        #-GRU layers-
        rl1=GRU(units=n_units1,return_sequences=True)(embed)
        rl2=GRU(units=n_units2)(rl1)        
        #-GRU layers-
        
        #-dense layer-
        dense1=Dense(n_dense,activation='relu')(rl2)
        #-dense layer-
        
        #-output-
        out = Dense(6,activation='sigmoid')(dense1)
        #-output-
        
        self.model = Model(inputs=inputs,outputs=out)
        
class gru_glove_model(KerasModels):
    
    def __init__(self,tox,modelParams={'embed_dim':100,'n_dense':50,
                                       'n_units1':100,'n_units2':100,
                                       'glove_embed_dim':50,
                                       'textType':'nw_excluded'}):
    
        '''
        This function initialize a neural netwok model with the 
        following architecture:
            The input is sequence of words. The input is fed into a network
            which is similar to the network used in `gru_model`.
            The difference is that here the input sequences 
            are also passed to an embeddig layer whose weights
            are obtained from Glove pre-trained model. The output of this
            layer is augmented with the output of the secong GRU
            layer and then is fed to the fully connected layers.
            
            see: 
            https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras
            -model.html
             
            
        Parameters
        ----------  
        tox: base.Toxic
            An object of class base.Toxic.
        modelParams: dict
            it must contain the following fields:
                `embed_dim`: dimension of the output of emebdding layer,
                `n_units1`: number of neurons in the first GRU layer.
                `n_units2`: number of neurons in the second GRU layer.
                'n_dense': number of neurons of the dense layer.
                `glove_embed_dim`: embedding dimension of glove features.
                `textType`: see base.Toxic.loadData() for details.
        '''
        self.model=None
        self.tox=tox
        self.__buildModel__(modelParams)  
        
    def __buildModel__(self,modelParams):
        '''
        This function builds the model.
        '''
        
        #===model parameters===
        embed_dim,n_units1,n_units2,n_dense,glove_embed_dim,textType=\
            (modelParams['embed_dim'],
             modelParams['n_units1'],
             modelParams['n_units2'],
             modelParams['n_dense'],
             modelParams['glove_embed_dim'],
             modelParams['textType'])
        ext={'raw':'raw','sw_excluded':'sw',
             'nw_excluded':'nw'}[textType]
        #===model parameters===
        
        #===build model===
        #-input layer-
        max_seq_len=self.tox.trainFeatureMat.shape[1]
        dict_size=np.max((self.tox.trainFeatureMat.max(),
                          self.tox.testFeatureMat.max()))
        inputs = Input(shape=(max_seq_len,))
        #-input layer-

        #-glove embedding-
        #load embedding weights
        fn=join(self.tox.dataDir,'embed_weights_dictsize_{}_d_{}_{}'.\
                format(dict_size+1,glove_embed_dim,ext))
        weights=np.fromfile(fn).reshape((-1,glove_embed_dim))
        glove_embed = Embedding(input_dim=weights.shape[0],
                                output_dim=glove_embed_dim,
                                input_length=max_seq_len,
                                trainable=False,weights=[weights])(inputs)
        #-glove embedding-
        
        #-trainable embedding layer-
        embed = Embedding(input_dim=dict_size+1,output_dim=embed_dim,
                          input_length=max_seq_len)(inputs)
        #-trainable embedding layer-

        #-GRU layers-
        rl1=GRU(units=n_units1,return_sequences=True)(embed)
        rl2=GRU(units=n_units2)(rl1)        
        #-GRU layers-
        
        #-augment glove and gru features-
        feat = concatenate([rl2,Flatten()(glove_embed)])
        #-augment glove and gru features-
        
        #-dense layer-
        dense1=Dense(n_dense,activation='relu')(feat)
        #-dense layer-
        
        #-output-
        out = Dense(6,activation='sigmoid')(dense1)
        #-output-
        
        self.model = Model(inputs=inputs,outputs=out)        
        
#=======================
#decision tree models
#=======================  
class randomForestModel(decisionTreeModel):    
    def __buildModel__(self,modelParams):
        
        #note that RandomForestClassifier supports multi-label 
        #classification.
        self.model=RandomForestClassifier(n_jobs=-1,**modelParams)
        

class lgbModel(decisionTreeModel):   
    def __buildModel__(self,modelParams):
        
        #note that lgbm does not supports multi-output classification.
        self.model=MultiOutputClassifier(LGBMClassifier(n_jobs=-1,
                                                        **modelParams))
        