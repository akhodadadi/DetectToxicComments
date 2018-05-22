import pandas as pd
from os.path import join    

dataDir = '/home/arash/MEGA/MEGAsync/'+\
'Machine Learning/Kaggle/Toxic/Data/'

df1=pd.read_csv(join(dataDir,'submission_model1_cnn_seqlen_300_dicsize_20000.csv'))
df2=pd.read_csv(join(dataDir,'submission_model2_gru_seqlen_300_dicsize_20000.csv'))
df3=pd.read_csv(join(dataDir,'submission_model1_cnn.csv'))
df4=pd.read_csv(join(dataDir,'submission_model2_gru.csv'))
df5=pd.read_csv(join(dataDir,'submission_model3_stack_seqlen_300_dicsize_20000.csv'))
df6=pd.read_csv(join(dataDir,'submission_model6_RF.csv'))


#Y=(df1.iloc[:,1:].values+df2.iloc[:,1:].values+df3.iloc[:,1:].values+\
#   df4.iloc[:,1:].values+df5.iloc[:,1:].values+df6.iloc[:,1:].values)/6

Y = .1*df1.iloc[:,1:].values+\
    .1*df2.iloc[:,1:].values+\
    .1*df3.iloc[:,1:].values+\
    .1*df4.iloc[:,1:].values+\
    .3*df5.iloc[:,1:].values+\
    .3*df6.iloc[:,1:].values 
   
df1.iloc[:,1:]=Y
df1.to_csv(join(dataDir,'submission_avg.csv'),index=False)



