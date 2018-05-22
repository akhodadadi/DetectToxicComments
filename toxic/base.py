from . import toxic_config
from os.path import join
from time import ctime
import pandas as pd
import re
from nltk.corpus import stopwords
sw = stopwords.words('english')
import numpy as np

#keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
    def __init__(self):
        self.dataDir=toxic_config.DATADIR
        self.classNames=toxic_config.CLASSES_NAME
        self.trainData=[];self.testData=[];
    
    def loadData(self,loadTrain=True,loadTest=True,textType='raw'):
        '''
        This function loads train and test data pandas.DataFrame.
        
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
        
        if (len(self.trainData)==0) or (len(self.trainData)==0):            
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
        
        if (len(self.trainData)==0) or (len(self.trainData)==0):            
            self.loadData(textType=textType)
            
        if loadOrCompute=='compute':
            Y = np.array(self.trainData.iloc[:,2:],dtype='int8')
            
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
            ext={'raw':'raw','sw_excluded':'sw',
                 'nw_excluded':'nw'}[textType]
            s='_seqlen_{}_dicsize_{}_{}'.format(max_seq_len,dict_size,ext)
            
            self.trainFeatureMat.tofile(join(self.dataDir,'X_train'+s))
            self.testFeatureMat.tofile(join(self.dataDir,'X_test'+s))
            Y.tofile(join(self.dataDir,'Y'+s))
            #---save to file---
        
