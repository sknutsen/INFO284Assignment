#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Obligatory assignment 1
# Students: skn003, tre081

import numpy as np
import pandas as pd
import re
import regex
import nltk
import matplotlib.pyplot as plt
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Storing the training dataset for further use
data = pd.read_csv('train.csv', delimiter = ',')

pd.options.display.max_colwidth = 100

display(data.head(4))


# In[2]:


# Printing the dimensions of the dataset as a reference point for upcoming trimming
print(data.shape)


# In[3]:


# Removing empty rows
filter = data['text'].str.contains('', regex = True, na = False)

data = data[filter]
print(data.shape)


# In[4]:


# Creating a new column with the combined data in the title, author and text columns
data['total']=data['title']+' '+data['author']+data['text']


# In[5]:


data=data.fillna('')


# In[6]:


filter = []

# Removing all rows with non-latin characters
for index, row in data.iterrows():
    reg = regex.match(r'[\p{Latin}\p{posix_punct}]+', row['total'])
    if reg == None:
        filter.append(False)
    else:
        filter.append(True)


# In[7]:


filter = np.array(filter)

filter


# In[8]:


# Applying filter to the dataset
data = data[filter]

# Printing results of the filter
print("after non-latin removal:", data.shape)


# In[9]:


# Removing newline characters
data['text'].replace(regex=True, inplace=True,to_replace='\n', value='')

# Exporting the dataset to csv format for storage
data.to_csv('newtrain.csv')


# In[10]:


# Creating a list with only the texts and none of the metadata
texts = data['text'].tolist()


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

# Building a vocabulary from all the texts
vectorizer.fit(texts)
vocabulary = vectorizer.vocabulary_ # Storing vocabulary in new variable for easier use
print("Before stopwords removal:", len(vocabulary)) # Printing the size of the vocabulary prior to removal of stopwords


# In[12]:


#import nltk
#nltk.download('stopwords')

# Retrieving a list of English stopwords
stopWords = set(stopwords.words('english'))

# Creating a copy of vocabulary for iteration
temp_dict = dict(vocabulary)
for key in temp_dict:
    if key in stopWords:
        del vocabulary[key] # Removing English stopwords from actual vocabulary


# In[13]:


# Printing results of the stopword removal
print("After stopwords removal:", len(vocabulary))


# In[14]:


# Exporting the vocabulary to a csv file
pd.DataFrame([vocabulary]).to_csv('Vocabulary.csv', sep=',')


# In[15]:


temp_dict = dict(vocabulary).keys()
for key in temp_dict:
    if not re.match(r'[^\W\d]*$', key):
        del vocabulary[key]


print(len(vocabulary))


# In[16]:


pd.DataFrame([vocabulary]).to_csv('Vocabulary2.csv', sep=',')


# In[17]:


temp_dict = dict(vocabulary).keys()
for key in temp_dict:
    for c in ['_', ')', '(', '{', '}', '[', ']', '/', ',', '.']:
        if c in key:
            del vocabulary[key]


print(len(vocabulary))


# In[ ]:


temp_dict = dict(vocabulary).keys()
for key in temp_dict:
    count = 0
    for text in data['total']:
        count += text.count(key)
    
    if count < 100:
        del vocabulary[key]


# In[ ]:


print(len(vocabulary))


# In[ ]:




