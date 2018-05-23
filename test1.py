from toxic.base import Toxic
from toxic.models import cnn_model,randomForestModel
from toxic import toxic_config
from os.path import join

dict_size=20000
max_seq_len=50
model='RF'
dataDir=toxic_config.DATADIR
print('\014')

if model=='cnn':    
    s= '_seqlen_{}_dictsize_{}'.format(max_seq_len,dict_size)
    save_model_path=join(dataDir,'fittedModels/fittedModel_model1_cnn'+s)
    submissionsDir=join(dataDir,'submissions/model1_cnn'+s+'.csv')
    
    T= Toxic()
    T.loadOrComputeTextSeq(loadOrCompute='load',dict_size=dict_size,
                           max_seq_len=max_seq_len)
    cnn=cnn_model(T)
    #cnn.fit(save_model_path,epochs=1)
    cnn.predict(dstDir=submissionsDir,loadModel=True,
                modelPath=save_model_path)
    
elif model=='RF':
    s= '_dictsize_{}'.format(dict_size)
    save_model_path=join(dataDir,'fittedModels/fittedModel_RF'+s)
    submissionsDir=join(dataDir,'submissions/RF'+s+'.csv')
    
    #compute features
    T= Toxic()
    T.loadOrCompute_tfidf(loadOrCompute='load',dict_size=dict_size,
                          words_ngram_range=(1,1),chars_ngram_range=(3,4))
    
    #fit model
    modelParams={'n_estimators':100,'min_samples_split':50,
                 'class_weight':'balanced_subsample',
                 'max_features':'auto','verbose':10}
    rf = randomForestModel(T,modelParams)
    rf.fit(save_model_path)
    rf.predict(dstDir=submissionsDir,loadModel=True,
               modelPath=save_model_path+'.pkl')
    
    

