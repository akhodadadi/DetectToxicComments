from toxic.base import Toxic
from toxic import toxic_config

#===load packages===
import pandas as pd
from os.path import join
from scipy.sparse import load_npz
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,make_scorer
from sklearn.multioutput import MultiOutputClassifier
#===load packages===

#===load data===
dataDir = '/home/arash/datasets/Kaggle/Toxic'
X_train = load_npz(join(dataDir,'train_tfidf_dict_size_20000.npz'))
X_test = load_npz(join(dataDir,'test_tfidf_dict_size_20000.npz'))
Y_train=np.fromfile(join(dataDir,'Y_train'),'int8').reshape((-1,6))
#===load data===

#===grid search over alpha===
mnb = MultiOutputClassifier(MultinomialNB())
param_grid={'estimator__alpha': np.logspace(-3,1,10)}
clf = GridSearchCV(mnb,param_grid=param_grid,verbose=10,
                   scoring=make_scorer(roc_auc_score,needs_threshold=True))
clf.fit(X_train,Y_train)
df=pd.DataFrame(clf.cv_results_)
#===grid search over alpha===

#===re-train best model and make predictions on test data===
clf = clf.best_estimator_.fit(X_train,Y_train)
Y_test = np.hstack([x[:,[1]] for x in clf.predict_proba(X_test)])

T = Toxic()
T.loadData()
Y_test = pd.concat([T.testData[['id']],pd.DataFrame(Y_test)],axis=1)
Y_test.columns=['id']+toxic_config.CLASSES_NAME
Y_test.to_csv(join(dataDir,'submissions/NB_dictsize_20000.csv'),
              index=False)
#===re-train best model and make predictions on test data===
