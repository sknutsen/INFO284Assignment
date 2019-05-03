#!/usr/bin/env python
# coding: utf-8

# In[37]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex
import re
import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import sys
filename = sys.argv[-1]

#Load data from csv-files
data = pd.read_csv(r'newtrain.csv', delimiter = ',')
test = open(filename, 'r').read()

filter = data['text'].str.contains('', regex = True, na = False)
data = data[filter]
data = data.fillna('')
stop_words = set(stopwords.words('english'))

#Create and apply filter for filtering out everything but english text
filter = list()
for index, row, in data.iterrows():
    reg = regex.match('^\p{Latin}', row['text'])
    if reg == None:
        filter.append(True)
    else:
        filter.append(False)
filter = np.array(filter)
    
#Make all titles and texts lowercase
data['text'] = data['text'].apply(lambda x: x.lower())
data['title'] = data['title'].apply(lambda x: x.lower())

#Remove all newlines
data['text'].replace(regex = True, inplace = True, to_replace = r'\n', value = '')

#Remove everything but word-characters and whitespace
data['text'].replace(regex = True, inplace = True, to_replace = r'[^a-zA-Z\s]', value = '')
data['title'].replace(regex = True, inplace = True, to_replace = r'[^a-zA-Z\s]', value = '')

#Create vocabulary and format data
vocabulary = {}
formattedData = []
for index, row, in data.iterrows():
    text_tokens = row['text'].split()
    title_tokens = row['title'].split()
    
    try:
        vect = CountVectorizer(stop_words = stop_words)
        vect.fit(text_tokens)
        text_vocabulary = vect.vocabulary_
        vocabulary.update(vect.vocabulary_)
        
        vect = CountVectorizer()
        vect.fit(title_tokens)
        title_vocabulary = vect.vocabulary_
        formattedData.append(row)
    except ValueError:
        continue

data = pd.DataFrame(formattedData)

#Write dataframe to csv
#data.to_csv('formattedData.csv')

#Write vocabulary to csv
#pd.DataFrame(vocabulary).to_csv("vocab.csv", header=None, index=None)

class NaiveBayesClassifier(object):
    def __init__(self):
        self.prior = defaultdict(int)
        self.logprior = {}
        self.bigdoc = defaultdict(list)
        self.loglikelihoods = defaultdict(defaultdict)
        self.vocabulary = []
        
    def create_vocabulary(self, documents):
        vocabulary = set()
        
        for document in documents:
            for word in document.split(" "):
                vocabulary.add(word.lower())
        return vocabulary
    
    def count_words_in_class(self):
        counts = {}
        for c in list(self.bigdoc.keys()):
            documents = self.bigdoc[c]
            counts[c] = defaultdict(int)
            for document in documents:
                words = document.split(" ")
                for word in words:
                    counts[c][word] += 1
        return counts
    
    def train(self, training_set, training_labels, alpha=1):
        total_docs = len(training_set)
        
        self.vocabulary = self.create_vocabulary(training_set)
        
        for x, y in zip(training_set, training_labels):
            self.bigdoc[y].append(x)
        
        self.word_count = self.count_words_in_class()
        
        for c in [0, 1]:
            total_c = sum(training_labels == c)
            
            self.logprior[c] = np.log(total_c) / total_docs
            
            total_count = 0
            for word in self.vocabulary:
                total_count += self.word_count[c][word]
                
            for word in self.vocabulary:
                count = self.word_count[c][word]
                self.loglikelihoods[c][word] = np.log(count + alpha) / (total_count + alpha * len(self.vocabulary))
                
    def predict(self, test_doc):
        sums = {0: 0, 1: 0}
        for c in self.bigdoc.keys():
            sums[c] = self.logprior[c]
            words = test_doc.split(" ")
            for word in words:
                if word in self.vocabulary:
                    sums[c] += self.loglikelihoods[c][word]
        return sums
    
nbc = NaiveBayesClassifier()
nbc.train(data['text'], data['label'])
print(nbc.predict(test))

