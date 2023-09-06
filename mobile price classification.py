#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#uploading the data set
df=pd.read_csv('train.csv')
print(df)


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.duplicated().sum()


# In[6]:


df.nunique()


# In[7]:


sns.lineplot(x="price_range", y="ram", data=df)
plt.title("Effect of ram on price")
plt.show()


# sns.pointplot(y="int_memory", x="price_range", data=df)
# plt.title("Effect of intrenal memory on price")
# plt.show()

# In[8]:


x=list(df['three_g'].value_counts())
l=['Supports 3G','Does not support 3G']
plt.title('percentage of 3G supported phones')
plt.pie(x,labels=l,autopct='%1.1f%%',pctdistance=0.85,explode=[0.01,0.01])
plt.show()


# In[9]:


x=list(df['four_g'].value_counts())
l=['Supports 4G','Does not support 4G']
plt.title('percentage of 4G supported phones')
plt.pie(x,labels=l,autopct='%1.1f%%',pctdistance=0.85,explode=[0.01,0.01])
plt.show()


# In[10]:


sns.jointplot(x='mobile_wt',y='price_range',data=df,kind='kde')
plt.title("Mobile weight vs price range")
plt.show()


# In[11]:


sns.boxplot(x="price_range", y="battery_power", data=df)
plt.show()


# In[12]:


sns.lineplot(x="price_range", y="fc", data=df)
plt.title("Effect of front cam on price")
plt.show()


# # Splitting and training data

# In[13]:


X=df.drop('price_range',axis=1)
y=df['price_range']


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = SVC()
model.fit(X_train, y_train)


# In[16]:


X_val = scaler.transform(X_val)


# In[17]:


dt=pd.read_csv('test.csv')
dt = dt.drop('id', axis=1)
dt = scaler.transform(dt)


# In[18]:


val_predictions = model.predict(X_val)


# In[19]:


accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)


# In[20]:


test_predictions = model.predict(dt)
test_predictions


# In[21]:


plt.hist(test_predictions, color="green")
plt.show()

