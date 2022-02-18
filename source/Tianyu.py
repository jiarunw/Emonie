# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-toolsai.jupyter added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\..\..\..\..\..\..\AppData\Local\Temp\99accb71-397b-498e-ad64-b7b8986b1281'))
	print(os.getcwd())
except:
	pass
# %%
import math
import pandas as pd
import kerouz_CNN as kr
import Embedding as yy
import numpy as np
from keras.preprocessing.text import Tokenizer # https://keras-cn.readthedocs.io/en/latest/preprocessing/text/
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# %%
if __name__=='__main__':

    # adjustable parameter
    offset=10 # l1=l3=l2+offset. namely l1 refers to get length, l2 refers to average length, l3 refers to train length
    rm_symbols='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    embedding_dim=300


# %%
# read dataset and dictionary
data_train=pd.read_csv('../dataset/Train.csv')

X_train=data_train['TEXT'].values
Y_train=data_train['Label'].values


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



'''
以下有待商榷

Y_train[Y_train == 2] = 1
Y_train[Y_train == 4] = 1
Y_train[Y_train == 10] = 1
'''




#Y_123 = Y_train[:]


Y_train=to_categorical(Y_train,20)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

#data_test=pd.read_csv('../dataset/Test.csv')
   # X_test=data_test['TEXT'].values
import gensim
f='enwiki_20180420_nolg_100d.txt'
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
weight=model.wv.vectors

emoji_map = pd.read_csv('../dataset/Mapping.csv')


# %%
# remove special symbols and stopwords from train set
X_rm=yy.corpus_pre(X_train)

# segmentation
tokenizer = Tokenizer(filters=rm_symbols, split=" ", lower=True) # filters：filter symbols that need to be removed lower：convert to lowercase
tokenizer.fit_on_texts(X_rm) # Tokenizer read train set free of special symbols. Results are stored in tokenize handle.

# vectorize. fill in and truncation
l2 = math.ceil(sum([len(s.split(" ")) for s in X_rm])/len(X_rm)) # l2:average length
l1 = l2+offset #get length
X_pd,tokenizer = kr.toknz(X_rm, l1,tokenizer)


# %%
#Dict that allocate an id(integer) to every word
ind_dict=tokenizer.word_index

#Dict that allocate an word vector to every word
# lookup_dict=yy.dict_pre(f)

# generate weightMatrix according to dictionary
W=yy.lookup(ind_dict,model,embedding_dim)


# %%
# train
model=kr.model_training(len(ind_dict)+1, W, l2+offset, X_pd, Y_train, embed_dim=embedding_dim, epochs=4)
print(model.predict_classes(X_pd[1:13])) #test on some sentences in the train data set

# %% [markdown]
# ## Predict on test set

# %%
# Prediction on test set
X_test_rm = yy.corpus_pre(X_test)
X_test_pd,_ = kr.toknz(X_test_rm, l1,tokenizer)
label_test = model.predict_classes(X_test_pd)
#for i in range(500, 521, 1):
    #print(emoji_map['emoticons'][label_test[i]])
    #print(X_test[i])


# %%
label_test=to_categorical(label_test,20)


# %%
accuracy = 0
inaccuracy = []
for i in range(len(Y_test)):
    if (label_test[i,:] == Y_test[i,:]).all():
        accuracy += 1

accuracy = accuracy/len(Y_test)
        


# %%
accuracy
##%
# %%
from sklearn import metrics
print('accuracy: ', metrics.accuracy_score(Y_test, label_test))
# %%


# %%
#loss, accuracy = model.evaluate(X_pd, Y_train, verbose=1)
#print("Accuracy = %f  ;  loss = %f" % (accuracy, loss))

# %% [markdown]
# ## Predict on user input

# %%
'''
user_str = input("input your sentence:")   
#user_str = "I love you"
X_user = np.array([str(user_str)])
print(X_user[0])
'''


# %%
'''
X_user_rm = yy.corpus_pre(X_user)
X_user_pd,_ = kr.toknz(X_user_rm, l1,tokenizer)
label_user = model.predict_classes(X_user_pd)
print(emoji_map['emoticons'][label_user[0]])
print(X_user[0]) 
'''


# %%



# %%



