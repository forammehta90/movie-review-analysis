# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
#import nltk
#nltk.download('wordnet')

class KNN(object):
    
    def __init__(self):
        self.csr = []
        self.rating = []
        self.csr_testing = []
    
    #Preprocessing of Training Data.
    def preprocessing_training(self):

        with open('train.dat','r') as file:
            df = pd.DataFrame(l.split("\t") for l in file)

        stop_words = set(stopwords.words('english'))
        stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','with','does']) # remove it if you need punctuation
        stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards']
        stop_words += ['again', 'against', 'all', 'almost', 'alone', 'along']
        stop_words += ['already', 'also', 'although', 'always', 'am', 'among']
        stop_words += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
        stop_words += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
        stop_words += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
        stop_words += ['because', 'become', 'becomes', 'becoming', 'been']
        stop_words += ['before', 'beforehand', 'behind', 'being', 'below']
        stop_words += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
        stop_words += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
        stop_words += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
        stop_words += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
        stop_words += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
        stop_words += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
        stop_words += ['every', 'everyone', 'everything', 'everywhere', 'except']
        stop_words += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
        stop_words += ['five', 'for', 'former', 'formerly', 'forty', 'found']
        stop_words += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
        stop_words += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
        stop_words += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
        stop_words += ['herself', 'him', 'himself', 'his', 'how', 'however']
        stop_words += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
        stop_words += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
        stop_words += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
        stop_words += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
        stop_words += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
        stop_words += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
        stop_words += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
        stop_words += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
        stop_words += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
        stop_words += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
        stop_words += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
        stop_words += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
        stop_words += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
        stop_words += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
        stop_words += ['some', 'somehow', 'someone', 'something', 'sometime']
        stop_words += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
        stop_words += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
        stop_words += ['then', 'thence', 'there', 'thereafter', 'thereby']
        stop_words += ['therefore', 'therein', 'thereupon', 'these', 'they']
        stop_words += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
        stop_words += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
        stop_words += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
        stop_words += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
        stop_words += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
        stop_words += ['whatever', 'when', 'whence', 'whenever', 'where']
        stop_words += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
        stop_words += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
        stop_words += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
        stop_words += ['within', 'without', 'would', 'yet', 'you', 'your']
        stop_words += ['yours', 'yourself', 'yourselves']
        st = LancasterStemmer()
        for ind,doc in enumerate(df[1]):

            doc.replace(",", "").replace(".", "").replace("(","").replace(")","").replace("/","").replace("?","").replace('<',"")
            dummy = [i.replace(",", "").replace(".", "").replace("(","").replace(")","").replace("/","").replace("?","").replace('<',"")\
                    .replace("'","").replace('>','').replace('!','').replace('/','')
                for i in doc.lower().split() if i not in stop_words and len(i) >= 3]

            self.csr.append(([st.stem(i) for i in dummy if str(i).isalpha() and len(i) >= 3 ]))
            self.rating.append(df[0][ind])


    def preprocessing_testing(self):
    
        with open('test.dat','r') as file:
            df = pd.DataFrame(l for l in file)
#        print df[0]

        stop_words = set(stopwords.words('english'))
        stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','with','does']) # remove it if you need punctuation
        stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards']
        stop_words += ['again', 'against', 'all', 'almost', 'alone', 'along']
        stop_words += ['already', 'also', 'although', 'always', 'am', 'among']
        stop_words += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
        stop_words += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
        stop_words += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
        stop_words += ['because', 'become', 'becomes', 'becoming', 'been']
        stop_words += ['before', 'beforehand', 'behind', 'being', 'below']
        stop_words += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
        stop_words += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
        stop_words += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de']
        stop_words += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
        stop_words += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
        stop_words += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
        stop_words += ['every', 'everyone', 'everything', 'everywhere', 'except']
        stop_words += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
        stop_words += ['five', 'for', 'former', 'formerly', 'forty', 'found']
        stop_words += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
        stop_words += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
        stop_words += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
        stop_words += ['herself', 'him', 'himself', 'his', 'how', 'however']
        stop_words += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
        stop_words += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
        stop_words += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
        stop_words += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
        stop_words += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
        stop_words += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
        stop_words += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
        stop_words += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
        stop_words += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
        stop_words += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
        stop_words += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
        stop_words += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
        stop_words += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
        stop_words += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
        stop_words += ['some', 'somehow', 'someone', 'something', 'sometime']
        stop_words += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
        stop_words += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
        stop_words += ['then', 'thence', 'there', 'thereafter', 'thereby']
        stop_words += ['therefore', 'therein', 'thereupon', 'these', 'they']
        stop_words += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
        stop_words += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
        stop_words += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
        stop_words += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
        stop_words += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
        stop_words += ['whatever', 'when', 'whence', 'whenever', 'where']
        stop_words += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
        stop_words += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
        stop_words += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
        stop_words += ['within', 'without', 'would', 'yet', 'you', 'your']
        stop_words += ['yours', 'yourself', 'yourselves']
        st = LancasterStemmer()
        
        for ind,doc in enumerate(df[0]):

            doc.replace(",", "").replace(".", "").replace("(","").replace(")","").replace("/","").replace("?","").replace('<',"")
            dummy = [i.replace(",", "").replace(".", "").replace("(","").replace(")","").replace("/","").replace("?","").replace('<',"")\
                    .replace("'","").replace('>','').replace('!','').replace('/','')
                for i in doc.lower().split() if i not in stop_words and len(i) >= 3]

            self.csr_testing.append(([st.stem(i) for i in dummy if str(i).isalpha() and len(i) >= 3 ]))

            

    def classifyNames(self,docs,rating,docs_test, k):
        mat,idx = self.build_matrix(docs)
        mat_test = self.build_matrix_test(docs_test,idx)
        # since we're using cosine similarity, normalize the vectors
#        print mat,"before"
        mat = self.csr_l2normalize(mat)
#        print mat.shape[0],'*'
#        print mat_test,"before"
        mat_test = self.csr_l2normalize(mat_test)
#        print mat_test.shape[0],'&'

        for i in range(mat_test.shape[0]):
            self.classify(mat_test[i,:], mat, rating,k)

    def classify(self,x, train, clstr,k):
        r""" Classify vector x using kNN and majority vote rule given training data and associated classes
        """
        # find nearest neighbors for x
        dots = x.dot(train.T)
#        print dots,"#"
        sims = list(zip(dots.indices, dots.data))
        sims.sort(key=lambda x: x[1], reverse=True)
#        print sims,"@@"
        if len(sims) == 0:
                # could not find any neighbors
            if np.random.rand() > 0.5:
                with open("format.dat", "a") as myfile:
                    myfile.write('+1'+'\n')  
            else:
                with open("format.dat", "a") as myfile:
                    myfile.write('-1'+'\n')                                 

        tc = Counter(clstr[s[0]] for s in sims[:k]).most_common()
#        print tc,"$$"
        if sims[0][1] >= 0.5:
            with open("format.dat", "a") as myfile:
                myfile.write(clstr[sims[0][0]]+'\n')
        elif len(tc) < 2:
            with open("format.dat", "a") as myfile:
                myfile.write(tc[0][0]+'\n')    
        elif tc[0][1] > tc[1][1]:
            with open("format.dat", "a") as myfile:
                myfile.write(tc[0][0]+'\n')
        elif tc[0][1] < tc[1][1]:
            with open("format.dat", "a") as myfile:
                myfile.write(tc[1][0]+'\n')
        else:
            tc = defaultdict(float)
            for s in sims[:k]:
                tc[clstr[s[0]]] += s[1]
            with open("format.dat", "a") as myfile:
                myfile.write(sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0] +'\n')

        """
        if len(tc) < 2 or tc[0][1] > tc[1][1]:
            # majority vote
            return tc[0][0]
        # tie break
        tc = defaultdict(float)
        for s in sims[:k]:
            tc[clstr[s[0]]] += s[1]
        return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]
        """

    def csr_l2normalize(self,mat):
        nrows = mat.shape[0]
        nnz = mat.nnz
        ind, val, ptr = mat.indices, mat.data, mat.indptr
        # normalize
        for i in range(nrows):
            rsum = 0.0    
            for j in range(ptr[i], ptr[i+1]):
                rsum += val[j]**2
            if rsum == 0.0:
                continue  # do not normalize empty rows
            rsum = 1.0/np.sqrt(rsum)
            for j in range(ptr[i], ptr[i+1]):
                val[j] *= rsum
        
        return mat

    def build_matrix_test(self,docs_test,idx):
        """ 
            Build sparse matrix from a list of documents, 
            each of which is a list of word/terms in the document.  
        """
        ncols = len(idx)
        nrows = len(docs_test)
        nnz = 0
        for d in docs_test:
            nnz += len(set(d))

    # set up memory
        ind = np.zeros(nnz, dtype=np.int)
        val = np.zeros(nnz, dtype=np.double)
        ptr = np.zeros(nrows+1, dtype=np.int)
        i = 0  # document ID / row counter
        n = 0  # non-zero counter
    # transfer values
        for d in docs_test:
            cnt = Counter(d)
            keys = list(k for k,_ in cnt.most_common())
            l = len(keys)
            for j,k in enumerate(keys):
                if k in idx:
                    ind[j+n] = idx[k]
                    val[j+n] = cnt[k]
            ptr[i+1] = ptr[i] + l
            n += l
            i += 1
            
        mat_test = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
        mat_test.sort_indices()
    
        return mat_test  
    
    def build_matrix(self,docs):
        """ 
            Build sparse matrix from a list of documents, 
            each of which is a list of word/terms in the document.  
        """
        nrows = len(docs)
        idx = {}
        tid = 0
        nnz = 0
        for d in docs:
            nnz += len(set(d))
            for w in d:
                if w not in idx:
                    idx[w] = tid
                    tid += 1
        ncols = len(idx)


    # set up memory
        ind = np.zeros(nnz, dtype=np.int)
        val = np.zeros(nnz, dtype=np.double)
        ptr = np.zeros(nrows+1, dtype=np.int)
        i = 0  # document ID / row counter
        n = 0  # non-zero counter
    # transfer values
        for d in docs:
            cnt = Counter(d)
            keys = list(k for k,_ in cnt.most_common())
            l = len(keys)
            for j,k in enumerate(keys):
                ind[j+n] = idx[k]
                val[j+n] = cnt[k]
            ptr[i+1] = ptr[i] + l
            n += l
            i += 1
            
        mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
        mat.sort_indices()
    
        return mat,idx



         
s = KNN()
s.preprocessing_training()
s.preprocessing_testing()
s.classifyNames(s.csr, s.rating,s.csr_testing, 31)