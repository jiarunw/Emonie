
#%%
from gensim.models import Word2Vec
import gensim
import pandas as pd
from keras.utils import to_categorical
import Embedding as em
from keras.preprocessing.text import Tokenizer
import numpy as np
import kerouz_CNN as mo
import math
import tensorflow as tf
import os
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#%%
embedding_dim=300
offset=2
# %%
# read dataset and dictionary
from sklearn.model_selection import train_test_split
data_train=pd.read_csv('../dataset/Train.csv')
X_train=data_train['TEXT'].values
Y_train=data_train['Label'].values
#%%
#reduce classes

#Y_train[Y_train == 5] = 1
Y_train[Y_train == 6] = 1
Y_train[Y_train == 11] = 1
Y_train[Y_train == 12] = 1
Y_train[Y_train == 13] = 1

Y_train[Y_train == 7] = 2
Y_train[Y_train == 15] = 2
Y_train[Y_train == 8] = 2
Y_train[Y_train == 14] = 2
Y_train[Y_train == 18] = 2
Y_train[Y_train == 9] = 2

#Y_train[Y_train == 0] = 4
Y_train[Y_train == 16] = 4
Y_train[Y_train == 19] = 4

    
Y_train[Y_train == 17] = 10
#%%

emoji_map = pd.read_csv('../dataset/Mapping.csv')
data_test=pd.read_csv('../dataset/Test.csv')
X_test=data_test['TEXT'].values

#%%

X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train, test_size=0.2, random_state=24)
# remove special symbols and stopwords from train set
X_rm=em.corpus_pre(X_train)

# segmentation
rm_symbols='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = Tokenizer(filters=rm_symbols, split=" ", lower=True) # filters：filter symbols that need to be removed lower：convert to lowercase
tokenizer.fit_on_texts(X_rm) # Tokenizer read train set free of special symbols. Results are stored in tokenize handle.
l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm))
X_pd,tokenizer = mo.toknz(X_rm, l2+offset,tokenizer)
ind_dict=tokenizer.word_index
# %%
X_seq=[]
for sentence in X_rm:
    words=list(sentence.lower().split())
    X_seq.append(words)
#%%
model = Word2Vec(sentences=X_seq, size=embedding_dim, window=2, min_count=1, workers=8)
model.save("word2vec.model")
#%%
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#%%
weight=model.wv.vectors
Y_trainC=to_categorical(Y_train,20)
l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm))
#%%
model=mo.model_training(len(weight), weight, l2+offset, X_pd, Y_trainC, embed_dim=embedding_dim, epochs=5)
#%%
print(model.predict_classes(X_pd[1:13])) #test on some sentences in the train data set
# %%
# Prediction on test set
X_test_rm = em.corpus_pre(X_test)
X_test_pd,_ = mo.toknz(X_test_rm, l2,tokenizer)
label_test = model.predict_classes(X_test_pd)
for i in range(5000,5100, 1):
    print(emoji_map['emoticons'][label_test[i]])
    print(X_test[i])

# %% 
for ii in range(5):
    user_str = input("input your sentence:")   
    #user_str = "I love you"
    X_user = np.array([str(user_str)])
    print(X_user[0])
    X_user_rm = em.corpus_pre(X_user)
    X_user_pd,_ = mo.toknz(X_user_rm, l2,tokenizer)
    label_user = model.predict_classes(X_user_pd)
    print(emoji_map['emoticons'][label_user[0]])
    print(X_user_rm[0]) 
# %%
from sklearn import metrics
print('accuracy: ', metrics.accuracy_score(Y_test, label_test))
# %%
