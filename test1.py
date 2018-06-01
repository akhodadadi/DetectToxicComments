from toxic.base import Toxic
from toxic.models import cnn_model,randomForestModel,lgbModel,gru_model,\
ensembleModels,gru_glove_model
from toxic import toxic_config
from os.path import join

dict_size=20000
max_seq_len=50
model='ensemble'
dataDir=toxic_config.DATADIR
print('\014')


if model=='cnn':
    modelParams={'embed_dim':100,'n_featMap':500,'kernel_size':[3,4,5],
                 'strides':[1]*3,'d_r':.1}
    
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

elif model=='gru':
    modelParams={'embed_dim':100,'n_dense':50,'n_units1':100,'n_units2':100}
    
    s= '_seqlen_{}_dictsize_{}'.format(max_seq_len,dict_size)
    save_model_path=join(dataDir,'fittedModels/fittedModel_gru'+s)
    submissionsDir=join(dataDir,'submissions/gru'+s+'.csv')
    
    T= Toxic()
    T.loadOrComputeTextSeq(loadOrCompute='load',dict_size=dict_size,
                           max_seq_len=max_seq_len)
    mdl=gru_model(T,modelParams)
    mdl.fit(save_model_path,epochs=10,patience=1)
    mdl.predict(dstDir=submissionsDir,loadModel=True,
                modelPath=save_model_path)
    
elif model=='gru_glove':
    modelParams={'embed_dim':100,'n_dense':50,'n_units1':100,'n_units2':100,
                 'glove_embed_dim':50,'textType':'nw_excluded'}
    
    s= '_seqlen_{}_dictsize_{}'.format(max_seq_len,dict_size)
    save_model_path=join(dataDir,'fittedModels/fittedModel_gru_glove'+s)
    submissionsDir=join(dataDir,'submissions/gru_glove'+s+'.csv')
    
    T= Toxic()
#    T.computeGlove(50,glovePath='/home/arash/datasets/glove',
#                   dict_size=dict_size,textType='nw_excluded')
    T.loadOrComputeTextSeq(loadOrCompute='load',dict_size=dict_size,
                           max_seq_len=max_seq_len)
    mdl=gru_glove_model(T,modelParams)
    mdl.fit(save_model_path,epochs=10,patience=1)
    mdl.predict(dstDir=submissionsDir,loadModel=True,
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
    
elif model=='lgbm':
    
    s= '_dictsize_{}'.format(dict_size)
    save_model_path=join(dataDir,'fittedModels/fittedModel_lgbm'+s)
    submissionsDir=join(dataDir,'submissions/lgbm'+s+'.csv')
    
    #compute features
    T= Toxic()
    T.loadOrCompute_tfidf(loadOrCompute='load',dict_size=dict_size,
                          words_ngram_range=(1,1),chars_ngram_range=(3,4))
    
    #fit model
    modelParams = {'num_leaves':31,'learning_rate':.1,
                   'subsample':.9,'colsample_bytree':.9,'reg_alpha':1.,
                   'objective':'binary','n_estimators':5000,'silent':False,
                   'subsample_for_bin':200000,'objective':'binary'}
#    fitParams = {'early_stopping_rounds':5,'eval_metric':'auc'}
    fitParams = {'early_stopping_rounds':5}
    gbm = lgbModel(T,modelParams)
    gbm.fit(save_model_path,monitor_eval=True,fitParams=fitParams)
    gbm.predict(dstDir=submissionsDir,loadModel=True,
                modelPath=save_model_path+'.pkl')
    
    
elif model=='ensemble':
    weights=[3,6,3,0,3]
    modelsOutDir='/home/arash/datasets/Kaggle/Toxic/submissions'
    filenames=['gru_seqlen_50_dictsize_20000.csv',
               'lgbm_dictsize_20000.csv',
               'model1_cnn_seqlen_50_dicsize_20000.csv',
               'RF_dictsize_20000.csv',
               'gru_glove_seqlen_50_dictsize_20000.csv']
    ensembleModels(modelsOutDir,filenames,weights)
    
    

                             
    
    

