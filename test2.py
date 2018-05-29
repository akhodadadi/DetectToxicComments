from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np

gbm=MultiOutputClassifier(LGBMClassifier(objective='binary',silent=False))

n_samples=100
X=np.random.randn(n_samples,4);Y=np.random.randint(0,3,100)
Y=np.random.randint(0,2,(n_samples,1))

gbm.fit(X,Y,sample_weight=None,estimatorParam={'verbose':True,
                                               'eval_set':[(X,Y)]})
y_pred=gbm.predict(X)