from . import toxic_config
from os.path import join,exists
from time import ctime
import pandas as pd
import re
from nltk.corpus import stopwords
sw = stopwords.words('english')
import numpy as np
from scipy.sparse import save_npz,hstack,load_npz
import gc

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

#keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input,Embedding,AveragePooling1D,Flatten
from keras.models import Model

def cleanText(text,removeStopWords=True):
    '''
    this function removes extra characters and stop-words.
    '''
        
    text = re.sub('[\W]',' ',text)
    if removeStopWords:
        tokens = filter(lambda x:(x.lower() not in sw) and (len(x)>2),
                        text.split())
        return ' '.join(tokens)
    else:
        return text

class Toxic():
    ''' 
    This class helps loading and cleaning data, feature extraction
    and visualization for Toxic dataset. It is also used as an argument
    of the `toxic.models` classes.
    '''
    def __init__(self):
        self.dataDir=toxic_config.DATADIR
        self.classNames=toxic_config.CLASSES_NAME
        self.trainData=[];self.testData=[];
        self.trainFeatureMat=[];self.testFeatureMat=[];self.Y_train=[]
    
    def loadData(self,loadTrain=True,loadTest=True,textType='raw'):
        '''
        This function loads train and test data into pandas.DataFrame.
        
        Parameters
        ---------
        loadTrain: bool
            if load train data.
        loadTest: bool
            if load test data.        
        textType: str
            one of 'raw' (load raw data),'sw_excluded' (load cleaned data
            where non-words and stop words have been removed),
            'nw_excluded' (load cleaned data where only non-words have been 
            removed).
        '''        
        
        print(ctime()+'...loading data into pandas df...')
        file_ext={'raw':'','sw_excluded':'_sw_excluded',
                  'nw_excluded':'_nw_excluded'}[textType]
        if loadTrain:
            filename=join(self.dataDir,'train{}.csv'.format(file_ext))
            self.trainData=pd.read_csv(filename)
        if loadTest:
            filename=join(self.dataDir,'test{}.csv'.format(file_ext))
            self.testData=pd.read_csv(filename)    
        
    def cleanData(self,removeStopWords=True):
        '''
        This function cleans the raw data. If self.trainData or 
        self.testData are empty it first calls self.loadData.
        It then removes either both the stop words and the non-words
        or only the non-words and saves the results as csv files.
        
        Parameters
        ----------
        removeStopWords: bool
            If True removes both stop-words and non-words, if False
            removes only non-words.
        '''
        
        if (len(self.trainData)==0) or (len(self.testData)==0):            
            self.loadData()
            
        if removeStopWords:
            file_ext='sw_excluded'#stop words and non-words excluded
        else:
            file_ext='nw_excluded'#only non-words excluded    
        
        #===clean train data===
        print(ctime()+'...cleaning train data...')
        self.trainData['comment_text'] = self.trainData['comment_text'].\
                        apply(lambda x:cleanText(x,removeStopWords))
        print(ctime()+'...saving train data...')
        filename=join(self.dataDir,'train_{}.csv'.format(file_ext))
        self.trainData.to_csv(filename,index=False)
        #===clean train data===

        #===clean test data===
        print(ctime()+'...cleaning test data...')
        self.testData['comment_text'] = self.testData['comment_text'].\
                        apply(lambda x:cleanText(x,removeStopWords))
        print(ctime()+'...saving test data...')
        filename=join(self.dataDir,'test_{}.csv'.format(file_ext))
        self.testData.to_csv(filename,index=False)
        #===clean train data===
        
    def loadOrComputeTextSeq(self,dict_size,max_seq_len,
                             loadOrCompute='compute',textType='nw_excluded'):
        '''
        This function either loads or computes the arrays represting
        text sequences. The result is a numpy.array of size 
        N x max_seq_len (where N is the sample size),
        with each row represeting the sequence of words 
        (integers in the range [0-dict_size]). The resulting array
        can be used as the input of keras models.
        
        If self.trainData or self.testData are empty it first calls
        self.loadData.
        
        Parameters
        ----------
        dict_size: int
            dictionary size: the maximum number of words to keep, based
            on word frequency. Only the most common dict_size words will
            be kept. This will be passed to 
            keras.preprocessing.text.Tokenizer
        max_seq_len: int
            maximum length of the sequence of words
        loadOrCompute: str
            either 'compute' or 'load'
        textType: str
            one of 'raw' (load raw data),'sw_excluded' (load cleaned data
            where non-words and stop words have been removed),
            'nw_excluded' (load cleaned data where only non-words have been 
            removed).            
            
        '''
        
            
        ext={'raw':'raw','sw_excluded':'sw',
             'nw_excluded':'nw'}[textType]
        s='_seqlen_{}_dictsize_{}_{}'.format(max_seq_len,dict_size,ext)
            
        if loadOrCompute=='compute':
            #---load data if it is not loaded yet---
            if (len(self.trainData)==0) or (len(self.testData)==0):            
                self.loadData(textType=textType)
            #---load data if it is not loaded yet---
            
            #---target values---
            self.Y_train = np.array(self.trainData.iloc[:,2:],dtype='int8')
            #---target values---
            
            #---tokenizing the texts---
            print(ctime()+'...tekenization...')
            T=Tokenizer(num_words=dict_size)
            train_texts=self.trainData.comment_text.tolist()
            test_texts=self.testData.comment_text.tolist()
            T.fit_on_texts(train_texts+test_texts)
            #---tokenizing the texts---
            
            #---convert tokenized text to sequence---
            print(ctime()+'...convert texts to sequeces...')
            train_seq=T.texts_to_sequences(train_texts)
            test_seq=T.texts_to_sequences(test_texts)
            #---convert tokenized text to sequence---
            
            #---convert sequences to array---
            print(ctime()+'...pad sequeces and convert to array...')
            self.trainFeatureMat = pad_sequences(train_seq,maxlen=max_seq_len)
            self.testFeatureMat = pad_sequences(test_seq,maxlen=max_seq_len)
            #---convert sequences to array---
            
            #---save to file---
            print(ctime()+'...saveing features to file...')            
            self.trainFeatureMat.tofile(join(self.dataDir,'X_train'+s))
            self.testFeatureMat.tofile(join(self.dataDir,'X_test'+s))
            self.Y_train.tofile(join(self.dataDir,'Y_train'))
            #---save to file---
            
        elif loadOrCompute=='load':
            fn=join(self.dataDir,'X_train'+s)
            self.trainFeatureMat=np.fromfile(fn,'int32').\
                                        reshape((-1,max_seq_len))
            fn=join(self.dataDir,'X_test'+s)
            self.testFeatureMat=np.fromfile(fn,'int32').\
                                        reshape((-1,max_seq_len))
            fn=join(self.dataDir,'Y_train')                                        
            self.Y_train=np.fromfile(fn,'int8').reshape((-1,6))
            
        
    def loadOrCompute_tfidf(self,dict_size,words_ngram_range=(1,1),
                            chars_ngram_range=(3,4),
                            loadOrCompute='compute',textType='sw_excluded'):
            
        if loadOrCompute=='compute':
            #---load data if it is notloaded yet---
            if (len(self.trainData)==0) or (len(self.testData)==0):            
                self.loadData(textType=textType)
            
            #missing texts
            self.trainData.fillna({'comment_text':'emptytext'},inplace=True)
            self.testData.fillna({'comment_text':'emptytext'},inplace=True)
            
            train_texts=self.trainData.comment_text.tolist()
            test_texts=self.testData.comment_text.tolist()
            self.Y_train = np.array(self.trainData.iloc[:,2:],dtype='int8')
            #---load data if it is notloaded yet---
            
            #---tfidf of words---
            print(ctime()+'...computing words tfidf...')
            tk = TfidfVectorizer(max_features=dict_size,                                    token_pattern=r'\w{1,}',
                                 stop_words='english',
                                 ngram_range=words_ngram_range,
                                 sublinear_tf=True,
                                 analyzer='word',)
            
            tk.fit(train_texts+test_texts)
            train_word_features = tk.transform(train_texts)
            test_word_features = tk.transform(test_texts)
            #---tfidf of words---
            
            #---tfidf of chars---
            print(ctime()+'...computing chars tfidf...')
            tk = TfidfVectorizer(max_features=dict_size,                                    token_pattern=r'\w{1,}',
                                 stop_words='english',
                                 ngram_range=chars_ngram_range,
                                 sublinear_tf=True,
                                 analyzer='char_wb')
            
            tk.fit(train_texts+test_texts)
            train_char_features = tk.transform(train_texts)
            test_char_features = tk.transform(test_texts)
            #---tfidf of chars---
            
            #---augment features---            
            self.trainFeatureMat=hstack([train_char_features,
                                         train_word_features])
            self.testFeatureMat=hstack([test_char_features,
                                        test_word_features])
            del train_char_features,test_char_features,
            train_word_features,test_word_features
            gc.collect()
            #---augment features---
            
            #---save to file---
            print(ctime() + '...saving to file...')
            fn=join(self.dataDir,'train_tfidf_dict_size_{}.npz'.\
                    format(dict_size))
            save_npz(fn,self.trainFeatureMat)
            fn=join(self.dataDir,'test_tfidf_dict_size_{}.npz'.\
                    format(dict_size))
            save_npz(fn,self.testFeatureMat)
            print(ctime() + '...DONE!')
            #---save to file---
            
        elif loadOrCompute=='load':
            print(ctime()+'...loading train data...')
            fn=join(self.dataDir,'train_tfidf_dict_size_{}.npz'.\
                    format(dict_size))
            self.trainFeatureMat=load_npz(fn)
            
            print(ctime()+'...loading test data...')
            fn=join(self.dataDir,'test_tfidf_dict_size_{}.npz'.\
                    format(dict_size))
            self.testFeatureMat=load_npz(fn)
            
            fn=join(self.dataDir,'Y_train')                                        
            self.Y_train=np.fromfile(fn,'int8').reshape((-1,6))
            
    def computeGlove(self,embed_dim,glovePath,dict_size=50000,
                     textType='nw_excluded'):
        '''
        This function computes and saves the embedding matrix
        corresponding to the Glove features.
        
        It will save the weights as numpy.array in self.dataDir.
        
        Parameters
        ---------
        embed_dim: int
            glove embedding dimension. It should be one of [50,100,200,300].
        glovePath: str
            path to the glove pre-trained weights.
        dict_size: int
            dictionary size
        textType: str 
            see `self.loadData` for details.
        '''
        
        if embed_dim not in [50,100,200,300]:
            msg='embed_dim should be one of the following:[50,100,200,300]'
            raise ValueError(msg)
        
        ext={'raw':'raw','sw_excluded':'sw',
             'nw_excluded':'nw'}[textType]
        
        #---load data if it is not loaded yet---
        if (len(self.trainData)==0) or (len(self.testData)==0):            
            self.loadData(textType=textType)
        #---load data if it is not loaded yet---
        
        #---target values---
        self.Y_train = np.array(self.trainData.iloc[:,2:],dtype='int8')
        #---target values---
        
        #---tokenizing the texts---
        print(ctime()+'...tekenization...')
        T=Tokenizer(num_words=dict_size)
        train_texts=self.trainData.comment_text.tolist()
        test_texts=self.testData.comment_text.tolist()
        T.fit_on_texts(train_texts+test_texts)
        #---tokenizing the texts---
        
        #---form a df containing words and their index and count---
        words_df=pd.DataFrame({'word':T.word_index.keys(),
                               'word_idx':T.word_index.values()})
        df=pd.DataFrame({'word':T.word_counts.keys(),
                         'word_count':T.word_counts.values()})
        words_df=words_df.merge(df,on='word')
        #---form a df containing words and their index and count---
        
        #---loading glove weights---
        print(ctime()+'...loading glove weights into pandas df...')
        fn=join(glovePath,'glove.6B.{}d.txt'.format(embed_dim))
        glove_df=pd.read_csv(fn,engine='python',delim_whitespace=True,
                             header=None)
        glove_df.rename({0:'word'},axis=1,inplace=True)
        #---loading glove weights---
        
        #---merge word_df and glove_df---
        words_df=words_df.merge(glove_df,on='word',how='left')
        words_df.fillna(0,inplace=True)
        words_df.sort_values(by='word_idx',inplace=True)
        embedding_weights=words_df.iloc[:,3:].values
        embedding_weights=np.vstack((np.zeros((1,embed_dim)),
                                     embedding_weights))
        #---merge word_df and glove_df---
        
        #---save the weights---
        print(ctime()+'...saving weights...')
        fn=join(self.dataDir,'embed_weights_dictsize_{}_d_{}_{}'.\
                format(dict_size,embed_dim,ext))
        embedding_weights.tofile(fn)
        #---save the weights---
        
        
    def loadOrComputeAvgGlove(self,embed_dim,dict_size,max_seq_len,
                              textType='nw_excluded',loadOrCompute='compute'):
        '''
        Loads or computes the average of the Glove 
        representation of words for each text in the corpus. So, each
        text will be represented by one vector which is obtained by
        averaging the Glove features of the words in the text.
        
        The method `computeGlove` should be called before this method
        is called. The method first checks if thweights exists in
        self.dataDir, if not it throws an error.
        
        The average glove feature is computed as follows: a neral
        network with one embedding layer and one output layer is built
        using Keras. The weights of the embedding layer are the Glove
        weights. The input to the network will be the texts' word sequences.
        The output is the average of the output of the embedding layer.
        Note that the output of the embedding layer has the shape
        (batch_size,max_seq_len,embed_dim) and we should compute 
        the average of the second dimension.
        
        Parameters
        ----------          
        embed_dim: int
            glove embedding dimension. It should be one of [50,100,200,300].
        dict_size: int
            vocabulary size
        max_seq_len: int
            maximum length of the sequence of words
        textType: str
            see `self.loadData` for details.
        '''
        
        if loadOrCompute=='compute':
            #---check if embed_dim has a legal value---
            if embed_dim not in [50,100,200,300]:
                msg='embed_dim should be one of the following:[50,100,200,300]'
                raise ValueError(msg) 
            #---check if embed_dim has a legal value---
            
            #---check if Glove weights file exist---
            ext={'raw':'raw','sw_excluded':'sw','nw_excluded':'nw'}[textType]        
            fn=join(self.dataDir,'embed_weights_dictsize_{}_d_{}_{}'.\
                    format(dict_size,embed_dim,ext))
            if not exists(fn):
                msg=\
                '''Glove weights file does not exists in {}. Please first \
                call self.computeGlove().
                '''.format(self.dataDir)
                raise ValueError(msg)
            #---check if Glove weights file exist---
            
            #---load Glove weights---
            fn=join(self.dataDir,'embed_weights_dictsize_{}_d_{}_{}'.\
                    format(dict_size,embed_dim,ext))
            weights=np.fromfile(fn).reshape((-1,embed_dim))       
            #---load Glove weights---
                
            #---load text sequences if it is not loaded yet---
            if (len(self.trainFeatureMat)==0) or (len(self.testFeatureMat)==0):
                self.loadOrComputeTextSeq(dict_size=dict_size,
                                          max_seq_len=max_seq_len,
                                          textType=textType,
                                          loadOrCompute='load')
            #---load text sequences if it is not loaded yet---
                    
            #---build a keras model with an embedding layer---
            inputs = Input(shape=(max_seq_len,))
            glove_embed = Embedding(input_dim=weights.shape[0],
                                    output_dim=embed_dim,
                                    input_length=max_seq_len,
                                    trainable=False,weights=[weights])(inputs)
            avgGlove = Flatten()(AveragePooling1D(pool_size=max_seq_len)\
                               (glove_embed))
            
            model = Model(inputs=inputs,outputs=avgGlove) 
            #---build a keras model with an embedding layer---
            
            #---compute average Glove---
            print('computing and saving average Glove features for:')
            
            print(ctime()+'...train data...')
            self.trainFeatureMat = model.predict(self.trainFeatureMat)
            fn=join(self.dataDir,'train_avgGlove_seqlen_dictsize_{}_d_{}'.\
                    format(max_seq_len,dict_size,embed_dim))
            self.trainFeatureMat.tofile(fn)
            
            print(ctime()+'...test data...')
            self.testFeatureMat= model.predict(self.testFeatureMat)
            fn=join(self.dataDir,'test_avgGlove_seqlen_dictsize_{}_d_{}'.\
                    format(max_seq_len,dict_size,embed_dim))
            self.testFeatureMat.tofile(fn)        
            #---compute average Glove---
            
        elif loadOrCompute=='load':
            #---load train features---
            fn=join(self.dataDir,'train_avgGlove_seqlen_dictsize_{}_d_{}'.\
                    format(max_seq_len,dict_size,embed_dim))
            self.trainFeatureMat=np.fromfile(fn,'float32').\
            reshape((-1,embed_dim))
            #---load train features---
            
            #---load test features---
            fn=join(self.dataDir,'test_avgGlove_seqlen_dictsize_{}_d_{}'.\
                    format(max_seq_len,dict_size,embed_dim))
            self.testFeatureMat=np.fromfile(fn,'float32').\
            reshape((-1,embed_dim))            
            #---load test features---
            
            #---load target values---
            fn=join(self.dataDir,'Y_train')                                        
            self.Y_train=np.fromfile(fn,'int8').reshape((-1,6))
            #---load target values---
            
        
        
        
        
        
        
