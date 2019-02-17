# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:26:40 2019

@author: soknu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv', delimiter = ',')
test = pd.read_csv('test.csv', delimiter = ',')

pd.options.display.max_colwidth = 100

display(data.head(4))

import re

print(data.shape)

filter = data['text'].str.contains('', regex = True, na = False)

data = data[filter]
print(data.shape)

data['total']=data['title']+' '+data['author']+data['text']

test=test.fillna('')
data=data.fillna('')

print("before non-latin removal:", data.shape)

import regex
filter = []
for index, row in data.iterrows():
    reg = regex.match('^\p{Latin}', row['text'])
    if reg == None:
        filter.append(True)
    else:
        filter.append(False)
        
filter

filter = np.array(filter)
filter

filter = np.array(filter)
print("after non-latin removal:", data.shape)

data['text'].replace(regex=True, inplace=True,to_replace='\n', value='')

import nltk

from nltk.stem import WordNetLemmatizer

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

true = data.loc[data['label'] == 0]
fake = data.loc[data['label'] == 1]

true.shape
fake.shape
