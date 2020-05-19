#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np


# In[90]:


def ReadData():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv").to_numpy()
    X_train = train_data.drop(columns = ["Active"]).to_numpy()
    y_train = train_data.drop(columns = ["Sequence"]).to_numpy()
    
    
    X_tr = np.array([np.array(X_train[i][0].replace("", " ").split(' ')[1:-1]) for i in range(X_train.shape[0])])
    X_test = np.array([np.array(test_data[i][0].replace("", " ").split(' ')[1:-1]) for i in range(test_data.shape[0])])
    
    
    return X_tr, y_train.ravel(), X_test

    #names = np.unique(X_tr)
    #gene_dict = dict(zip(names, range(names.shape[0])))
    
    #for i in range(X_tr.shape[1]):
        #X_tr[:,i]= np.array([gene_dict[x] for x in X_tr[:,i]])
        #X_test[:,i] = np.array([gene_dict[x] for x in X_test[:,i]])
    
    #return X_tr.astype(np.int), y_train.ravel(), X_test.astype(np.int)
    


# In[61]:


def WriteData(data,filename = 'submission'):
    pd.DataFrame(data).to_csv(filename + '.csv', index=False, header=False, sep=' ')


# In[ ]:




