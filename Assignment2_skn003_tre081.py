#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Obligatory assignment 2
# Students: skn003, tre081

import numpy as np
import pandas as pd
import re
import regex
import nltk
import matplotlib.pyplot as plt


# In[15]:


# Assignment 2 prep

# Reading vocabulary and dataset for future use
vocab = pd.read_csv('Vocabulary_final.csv', delimiter = ',')
train = pd.read_csv('newtrain.csv', delimiter = ',')


# In[17]:


temp_vocab = dict(vocab).keys()
new_vocab = {}
vocab_reliable = {}
vocab_unreliable = {}
labels = train['label']
totals = train['total']
for word in temp_vocab:
    count = 0
    unreliable = 0
    reliable = 0
    for i in range(0, 100):
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


# In[18]:


df_values = {'word': list(new_vocab.keys()), 'count': list(new_vocab.values())}
df = pd.DataFrame(data=df_values)
pd.DataFrame(df).to_csv('Vocabulary_with_counts.csv', sep=',')

df_ur_values = {'word': list(vocab_unreliable.keys()), 'count': list(vocab_unreliable.values())}
df_ur = pd.DataFrame(data=df_ur_values)
pd.DataFrame(df_ur).to_csv('Vocabulary_unreliable.csv', sep=',')

df_r_values = {'word': list(vocab_reliable.keys()), 'count': list(vocab_reliable.values())}
df_r = pd.DataFrame(data=df_r_values)
pd.DataFrame(df_r).to_csv('Vocabulary_reliable.csv', sep=',')


# In[22]:


# The classifier    
def classifier(inp):
    # Stores the full vocabulary with total word count
    vocab = pd.read_csv('Vocabulary_with_counts.csv', delimiter = ',') 
    # Stores the full vocabulary with number of times each word occurrs in reliable texts
    vocab_reliable = pd.read_csv('Vocabulary_reliable.csv', delimiter = ',') 
    # Stores the full vocabulary with number of times each word occurrs in unreliable texts
    vocab_unreliable = pd.read_csv('Vocabulary_unreliable.csv', delimiter = ',')
    vocab_labels = {}
    vocab_words = vocab['word']
    reliable_word_count = vocab_reliable['count']
    unreliable_word_count = vocab_unreliable['count']
    for i in range(len(vocab['word'])):
        # Determining whether a word is reliable and assigning a value consistent with training dataset
        if reliable_word_count[i] >= unreliable_word_count[i]:
            vocab_labels[vocab_words[i]] = 0
        else:
            vocab_labels[vocab_words[i]] = 1
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
        print(avg_reliability)
        return 1
    elif int(avg_reliability) == 0:
        print("This text is reliable!")
        print(avg_reliability)
        return 0


# In[23]:


'''
__Function__ 
testset_label_difference:
    When the classifier is run
    Increment incorrect_label by 1 for each incorrectly labelled text
'''
def testset_label_difference(testset, target):
    testset_length = len(testset)
    incorrect_label = 0
    
    '''
    __Function__ 
    error_rate_calculator:
        Divides number of incorrectly labelled texts by total number of texts
        Prints results
    '''
    def error_rate_calculator():
        print('Total length of test set:', testset_length)
        print('Number of incorrectly labelled texts:', incorrect_label)
        print('The error rate of the classifier is:', incorrect_label / testset_length)
        
    i = 0
    while i < testset_length:
        new_label = classifier(testset[i])  # Running the classifier on text in test set
        if target[i] != new_label:
            incorrect_label += 1  # Increase the count if the classifier is wrong
        i += 1
        
    error_rate_calculator()  # Calculating the results


# In[19]:


import sklearn
from sklearn.model_selection import train_test_split


# In[20]:


data_tar = pd.DataFrame(train['label'], columns = ['Unreliable'])


# In[21]:


X = pd.DataFrame(train['text'], columns = df_values['word'])
y = np.ravel(data_tar)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[24]:


# Testing the classifier on the test set to see the accuracy
testset_label_difference(X_test, y_test)


# In[ ]:




