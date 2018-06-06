In this project we design and compare several models for detecting different kinds of toxic comments in a dataset of comments from Wikipedia’s talk page edits. Each sample in the dataset includes a comment and 6 binary values which are human labels on the type(s) of toxicity of the comment. The toxicity types are: toxic, severe_toxic, obscene, threat, insult and identity_hate. A comment may have several labels.

We will develope three types of model: decision tree models (random forest, and boosted trees implemented in XGboost and lightGBM) trained on *term-frequency-inverse-document-frequency (tf-idf)* of the texts, a neural network with an embedding layer followed by a convolutional and dense layer, and a neural network with an embedding layer followed by two GRU layers and a dense layer. In addition, we will examine if using the *Glove* representation of the words as features will improve the perfomance of these models.

The results show that a gradient boosted tree (LightGBM) with tf-idf features achieved the best performance. Also ensembling the models by a simple weighting scheme improved the performance (measured by AUC on the test set) by about 2%.

Please read the full report [here](https://github.com/akhodadadi/DetectToxicComments/blob/master/Detecting%20toxic%20comments.ipynb).

