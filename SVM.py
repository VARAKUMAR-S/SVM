#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


letters = pd.read_csv("D:\\360DigiTMG\\Black Box Technique SVM\\STUDY MATERIAL\\letterdata.csv")
letters.describe()


# In[3]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[4]:


train,test = train_test_split(letters, test_size = 0.20)

train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]


# In[5]:


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)


# In[7]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)

