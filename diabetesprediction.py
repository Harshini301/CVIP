#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#uploading the data set
df=pd.read_csv('diabetes_prediction_dataset.csv')
print(df)


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace=True)
print(df)


# In[9]:


df.nunique()


# In[10]:


df['gender'].unique()


# In[11]:


df['hypertension'].unique()


# In[12]:


df['heart_disease'].unique()


# In[13]:


df['smoking_history'].unique()


# In[14]:


df['diabetes'].unique()


# In[15]:


x=list(df['diabetes'].value_counts())
print(df['diabetes'].value_counts())
l=['not daiabetic','diabetic']
plt.title('percentage of diabetics')
plt.pie(x,labels=l,autopct='%1.1f%%',pctdistance=0.85,explode=[0.01,0.01])
plt.show()


# In[16]:


df1 = df[df['diabetes']==1]
print(df1)


# In[17]:


#distribution in age
plt.hist(df1["age"], color="green")
plt.xlabel("age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.show()


# In[18]:


fig=plt.figure(figsize=(5,5))
sns.countplot(x=df['smoking_history'])
plt.title('analysing smoking history of daibetic people')
plt.xlabel('Smoking history')
plt.ylabel('count')
plt.show()


# In[19]:


#distribution of diabetics with respect to gender
sns.violinplot(x ="diabetes", y ="gender",data = df)
plt.show()


# In[20]:


sns.violinplot(x ="blood_glucose_level", y ="age",data = df1)
plt.title('distribution of sugar level among diabetic people with respect to age')
plt.show()


# In[21]:


df2 = df[df['diabetes']==0]
print(df2)


# In[22]:


sns.violinplot(x ="blood_glucose_level", y ="age",data = df2)
plt.title('distribution of sugar level among non-diabetic people with respect to age')
plt.show()


# In[23]:


df.hist(figsize=(20,20))
plt.show()


# In[24]:


df1.hist(figsize=(20,20))
plt.title("Analysis on people suffering from daibetics")
plt.show()


# In[25]:


df2.hist(figsize=(20,20))
plt.title("Analysis on people with no daibetics")
plt.show()


# In[26]:


dataplot=sns.heatmap(df.corr(),annot=True)
plt.show()


# # Model Building

# In[42]:


x=df.drop('diabetes',axis=1)
x=x.drop('gender',axis=1)
x=x.drop('smoking_history',axis=1)
y=df['diabetes']


# In[44]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape,y_train.shape


# In[40]:


x_test.shape,y_test.shape


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score


# In[45]:


reg = LogisticRegression()
reg.fit(x_train,y_train)       


# In[46]:


lr_pred=reg.predict(x_test)


# In[47]:


print("Classification Report is:\n",classification_report(y_test,lr_pred))
print("\n F1:\n",f1_score(y_test,lr_pred))
print("\n Precision score is:\n",precision_score(y_test,lr_pred))
print("\n Recall score is:\n",recall_score(y_test,lr_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,lr_pred))

