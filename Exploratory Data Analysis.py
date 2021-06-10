#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data

# In[2]:


train = pd.read_csv("titanic_train.csv")


# In[3]:


train.head()


# # Data Analysing

# Checking Missing Data

# In[4]:


train.isnull()


# In[5]:


ax = sns.heatmap(train.isnull(),cbar=False,cmap='viridis', yticklabels=False)


# Using heatmap we can see that age and cabin has lot of missing values.But in between these two cabin data is almost completely lost so we will drop that col and with age we can do some replacement with imputation.
# We can continue analysing the gathered data.

# In[6]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[16]:


sns.distplot(train['Age'].dropna(),kde=False,color="r",bins=40)


# In[17]:


train['Age'].hist(bins=30,color='g',alpha=0.3)


# In[18]:


sns.countplot(x='SibSp',data=train)


# In[19]:


train['Fare'].hist(color='blue',bins=40,figsize=(8,4))


# # Data Cleaning 

# In[21]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[26]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    


# In[27]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[28]:


train.head()


# Null Values in Age column is resolved

# In[29]:


ax = sns.heatmap(train.isnull(),cbar=False,cmap='viridis', yticklabels=False)


# droping the cabin column

# In[30]:


train.drop('Cabin',axis=1,inplace=True)


# after removing cabin col

# In[32]:


ax = sns.heatmap(train.isnull(),cbar=False,cmap='viridis', yticklabels=False)


# In[33]:


train.head()


# In[34]:


train.info()


# In[38]:


train.dropna(inplace=True)


# In[39]:


train.info()


# Converting Categorical Features

# In[40]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[41]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[43]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[44]:


train.head()


# In[45]:


train = pd.concat([train,sex,embark],axis=1)


# In[46]:


train.head()


# # Building a Logistic Regression Model

# # Train Test Split

# In[48]:


train.drop('Survived',axis=1).head()


# In[49]:


train['Survived'].head()


# In[50]:


from sklearn.model_selection import train_test_split


# In[52]:


X_train,X_test,y_train,y_test = train_test_split(train.drop('Survived',axis=1),
                                                 train['Survived'],test_size=0.30,
                                                 random_state=12)


# # Training and Predicting

# In[53]:


from sklearn.linear_model import LogisticRegression


# In[54]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[55]:


predictions = logmodel.predict(X_test)


# In[57]:


from sklearn.metrics import confusion_matrix


# In[59]:


accuracy = confusion_matrix(y_test,predictions)


# In[62]:


accuracy


# In[63]:


from sklearn.metrics import accuracy_score


# In[64]:


accuracy = accuracy_score(y_test,predictions)
accuracy


# In[66]:


predictions.shape


# In[67]:


predictions


# In[ ]:




