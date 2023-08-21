#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#uploading the data set
df=pd.read_csv('Covid Data.csv')
print(df)


# In[3]:


df.head(10)


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# # Data preprocessing

# In[7]:


df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace=True)
print(df)


# In[9]:


df.nunique()


# In[10]:


#finding whether person is died or alive
df.DATE_DIED[df['DATE_DIED'] != '9999-99-99'] = 'Died'
df.DATE_DIED[df['DATE_DIED'] == '9999-99-99'] = 'Alive'


# In[11]:


#renaming the column
df.rename(columns={'DATE_DIED':'Status'},inplace=True)
print(df)


# In[12]:


#people survival status
x=list(df['Status'].value_counts())
l=['Dead','Alive']
plt.pie(x,labels=l,autopct='%1.1f%%',pctdistance=0.85,explode=[0.01,0.01])
plt.show()


# In[13]:


#distribution in age
plt.hist(df["AGE"], color="green")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.show()


# In[14]:


#checking medication status
x=list(df['PATIENT_TYPE'].value_counts())
l=['Returned home','Medicated']
plt.title("Medication status")
plt.pie(x,labels=l,autopct='%1.1f%%',pctdistance=0.85,explode=[0.01,0.01],colors=['lightgreen','cyan'])
plt.show()


# In[15]:


fig=plt.figure(figsize=(5,5))
sns.countplot(x=df['ICU'])
plt.title('Count of patients admitted in ICU')
plt.xlabel('Admitted to ICU')
plt.ylabel('Patients count')
plt.show()


# In[16]:


fig=plt.figure(figsize=(5,5))
sns.countplot(x=df['PREGNANT'])
plt.title('Count of pregnent people')
plt.show()


# In[17]:


#deaths/lived with respect to age
sns.violinplot(x ="AGE", y ="Status",data = df)
plt.show()


# In[20]:


sns.violinplot(x ="Status", y ="SEX",data = df)
plt.show()


# In[22]:


plot = sns.FacetGrid(df, col="ASTHMA")
plot.map(plt.plot, "AGE")
plt.show()


# In[23]:


print(df.corr())


# In[29]:


fig=plt.figure(figsize=(15,15))
dataplot=sns.heatmap(df.corr())
plt.show()

