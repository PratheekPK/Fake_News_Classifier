#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


train=pd.read_csv(r'C:\Users\hp\Downloads\fake-news\train.csv')
test=pd.read_csv(r'C:\Users\hp\Downloads\fake-news\test.csv')


# In[3]:


train=train.fillna('')
test=test.fillna('')


# In[4]:


train['total'] = train['title']+' '+train['author']+' '+train['text']
test['total']=test['title']+' '+test['author']+' '+test['text']


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train_x=train["total"]
test_x=test["total"]
train_y=train["label"]


# In[8]:


vectoriser=TfidfVectorizer(stop_words="english",max_df=0.7)


# In[9]:


train_x.head()


# In[10]:


train_x=vectoriser.fit_transform(train_x.values.astype('U'))
test_x=vectoriser.transform(test_x.values.astype('U'))


# In[11]:


classifier=PassiveAggressiveClassifier(max_iter=50)
classifier.fit(train_x,train_y)
y_pred=classifier.predict(test_x)


# In[12]:


submission2 =pd.DataFrame()
submission2["id"]=test["id"]
submission2['label'] = y_pred
submission2.to_csv('submission.csv',index=False)


# In[13]:


submission2.head()


# In[ ]:




