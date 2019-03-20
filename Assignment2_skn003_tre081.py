#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Obligatory assignment 2
# Students: skn003, tre081

import numpy as np
import pandas as pd
import re
import regex
import nltk
import matplotlib.pyplot as plt


# In[7]:


# Assignment 2 prep

# Reading vocabulary and dataset for future use
vocabulary = pd.read_csv('Vocabulary.csv', delimiter = ',')
train = pd.read_csv('newtrain.csv', delimiter = ',')

temp_dict = dict(vocabulary).keys()
for key in temp_dict:
    if not re.match(r'[^\W\d]*$', key):
        del vocabulary[key]    # Deleting all words that contain numbers


print('Size of vocabulary after removing words containing numbers: ', len(vocabulary.keys()))


# In[8]:


temp_dict = dict(vocabulary).keys()
for key in temp_dict:
    for c in ['_', ')', '(', '{', '}', '[', ']', '/', ',', '.']:
        if c in key:
            del vocabulary[key]    # Deleting all strings that contain symbols


print('Size of vocabulary after removing words containing symbols: ', len(vocabulary.keys()))


# In[ ]:


temp_dict = dict(vocabulary).keys()
for word in temp_dict:
    count = 0
    for t in train['total']:
        count += t.count(word)    # Counting the ammount of times each word appears
        
    if count < 1000:
        del vocabulary[word]      # Deleting all words that occurr less than 1000 times


pd.DataFrame([vocabulary]).to_csv('Vocabulary_final.csv', sep=',')    # Saving vocabulary cleaning progress


# In[ ]:


vocab = pd.read_csv('Vocabulary_final.csv', delimiter = ',')    # Continuing vocabulary cleaning progress from file


temp_dict = dict(vocab).keys()
new_vocab = {}
vocab_reliable = {}
vocab_unreliable = {}
labels = data['label']
totals = data['total']
for word in temp_dict:
    count = 0
    unreliable = 0
    reliable = 0
    for i in range(0, len(labels)-1):
        try:
            wc = totals[i].count(word)     # Counting the ammount of times each word appears
            count += wc
            if labels[i] == 1:
                unreliable += wc           # If the text is unreliable then the "unreliable wordcount" is updated
            elif labels[i] == 0:
                reliable += wc             # If the text is reliable then the "reliable wordcount" is updated
        except KeyError:
            x = 1 + 1                      # Filler line
    new_vocab[word] = count                # Adding word with total amount of times it occurrs
    vocab_reliable[word] = reliable        # Adding word with amount of times it occurrs in reliable texts
    vocab_unreliable[word] = unreliable    # Adding word with amount of times it occurrs in unreliable texts


# In[ ]:


df_small_values = {'word': list(new_vocab.keys()), 'count': list(new_vocab.values())}
df_small = pd.DataFrame(data=df_small_values)
pd.DataFrame(df_small).to_csv('Vocabulary_small.csv', sep=',')

df_ur_values = {'word': list(vocab_unreliable.keys()), 'count': list(vocab_unreliable.values())}
df_ur = pd.DataFrame(data=df_ur_values)
pd.DataFrame(df_ur).to_csv('Vocabulary_small_unreliable.csv', sep=',')

df_r_values = {'word': list(vocab_reliable.keys()), 'count': list(vocab_reliable.values())}
df_r = pd.DataFrame(data=df_r_values)
pd.DataFrame(df_r).to_csv('Vocabulary_small_reliable.csv', sep=',')


# In[ ]:


train.head()


# In[ ]:


train = train.drop(columns=['total'])
train.head()


# In[ ]:


train.to_csv('newtrain_final.csv')


# In[ ]:


# The classifier    
def classifier(inp):
    # Stores the full vocabulary with total word count
    vocab = pd.read_csv('Vocabulary_small.csv', delimiter = ',') 
    # Stores the full vocabulary with number of times each word occurrs in reliable texts
    vocab_reliable = pd.read_csv('Vocabulary_small_reliable.csv', delimiter = ',') 
    # Stores the full vocabulary with number of times each word occurrs in unreliable texts
    vocab_unreliable = pd.read_csv('Vocabulary_small_unreliable.csv', delimiter = ',')
    vocab_labels = {}
    for word in vocab.keys():
        # Determining whether a word is reliable and assigning a value consistent with training dataset
        if vocab_reliable[word] >= vocab_unreliable[word]:
            vocab_labels[word] = 0
        else:
            vocab_labels[word] = 1
    text = ''
    # If the input is a filename we open and read the associated textfile
    if inp[-4:] == '.txt':
        file = open(inp, 'r')
        text = file.read()
        file.close()
    else:
        text = inp
    reliability_sum = 0
    total_count = 0
    for w in vocab_labels.keys():
        words = text.split()
        for word in words:
            clone = word
            for c in clone:
                if c in ['_', ')', '(', '{', '}', '[', ']', '/', ',', '.']:
                    word.replace(c, '')
            if w == word:
                reliability_sum += vocab_labels[w]
                total_count += 1
    avg_reliability = reliability_sum / total_count    # Calculating the average reliability of the text
    # Converting the average to integer so that it will either be 1, unreliable, or 0, reliable.
    if int(avg_reliability) == 1:
        print("This text is unreliable!")
    elif int(avg_reliability) == 0:
        print("This text is reliable!")


# In[ ]:





# In[ ]:




