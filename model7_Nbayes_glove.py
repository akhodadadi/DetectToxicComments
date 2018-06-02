from toxic.base import Toxic
from toxic import toxic_config

#===load packages===
from os.path import join
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
print('\014')
#===load packages===

#===load feature and target values===
tox = Toxic()
tox.loadOrComputeAvgGlove(embed_dim=50,dict_size=20000,max_seq_len=50,
                          loadOrCompute='load')
Y_train=np.fromfile(join(tox.dataDir,'Y_train'),'int8').reshape((-1,6))
#===load feature and target values===

#===build and fit the model===
gnb=MultiOutputClassifier(GaussianNB())
gnb=gnb.fit(tox.trainFeatureMat,Y_train)
#===build and fit the model===

#===make predictions on test data===
Y_test = np.hstack([x[:,[1]] for x in gnb.predict_proba(tox.testFeatureMat)])
tox.loadData()
Y_test = pd.concat([tox.testData[['id']],pd.DataFrame(Y_test)],axis=1)
Y_test.columns=['id']+toxic_config.CLASSES_NAME
Y_test.to_csv(join(tox.dataDir,'submissions/NB_glove_dictsize_20000.csv'),
              index=False)

#===make predictions on test data===