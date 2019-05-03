# Obligatory assignment 2
# Students: skn003, tre081
from collections import defaultdict

import numpy as np
import pandas as pd

# Assignment 2 prep

# Reading vocabulary and dataset for future use
vocabulary = pd.read_csv('Vocabulary_final.csv', delimiter=',')
train = pd.read_csv('newtrain.csv', delimiter=',')

temp_vocab = dict(vocabulary).keys()
new_vocab = {}
labels = train['label']
totals = train['total']
for word in temp_vocab:
    count = 0
    for i in range(0, 100):
        try:
            wc = totals[i].count(word)     # Counting the ammount of times each word appears
            count += wc
        except KeyError:
            x = 1 + 1                      # Filler line
    new_vocab[word] = count                # Adding word with total amount of times it occurrs

df_values = {'word': list(new_vocab.keys()), 'count': list(new_vocab.values())}
df = pd.DataFrame(data=df_values)
pd.DataFrame(df).to_csv('Vocabulary_with_counts.csv', sep=',')


# The classifier
class Classifier(object):
    def __init__(self):
        # Stores the full vocabulary with total word count
        self.vocab = pd.read_csv('Vocabulary_with_counts.csv', delimiter=',')
        self.vocab_labels = defaultdict(int)
        self.vocab_words = self.vocab['word']
        self.text_label = defaultdict(list)
        self.initial_probability = defaultdict(defaultdict)
        self.probabilities = defaultdict(defaultdict)

    def get_text_label(self):
        return self.text_label

    def predict(self, test):
        text = ''
        # If the input is a filename we open and read the associated textfile
        if str(test)[-4:] == '.txt':
            file = open(test, 'r')
            text = file.read()
            file.close()
        else:
            text = test

        self.vocab_words = list(self.vocab.keys())
        sums = {0: 0, 1: 0}
        for label in [0, 1]:
            sums[label] = self.initial_probability[label]
            words = str(text).split(" ")
            for w in words:
                if w in self.vocab_words:
                    sums[label] += self.probabilities[label][w]
        return sums

    def train(self):
        alpha = 0.1
        vocab_words = list(self.vocab.keys())
        word_counts_by_label = {}
        total_texts = len(train['text'])

        for t, l in zip(train['text'], train['label']):
            self.text_label[l].append(t)

        for k in [0, 1]:
            all_texts = self.text_label[k]
            word_counts_by_label[k] = defaultdict(int)
            for t in all_texts:
                words = str(t).split(" ")
                for w in words:
                    word_counts_by_label[k][w] += 1

        for label in [0, 1]:
            label_total_count = sum(train['label'] == label)
            self.initial_probability[label] = np.log(label_total_count / total_texts)
            total_count = 0
            for w in vocab_words:
                total_count += word_counts_by_label[label][w]

            for w in vocab_words:
                c = word_counts_by_label[label][w]
                self.probabilities[label][w] = np.log((c + alpha) / (total_count + alpha * len(self.vocab)))


'''
__Function__ 
testset_label_difference:
    When the classifier is run
    Increment incorrect_label by 1 for each incorrectly labelled text
'''
def testset_label_difference():
    classifier = Classifier()
    classifier.train()
    testset = train['text']
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

    for k in list(testset.keys()):
        print(k)
        for v in testset[k]:
            predict_result = classifier.predict(v)
            new_label = 0
            if predict_result[0] < predict_result[1]:
                new_label = 1  # Running the classifier on text in test set
            if k != new_label:
                incorrect_label += 1  # Increase the count if the classifier is wrong

    error_rate_calculator()  # Calculating the results


classifier = Classifier()
classifier.train()
result = classifier.predict(train['text'][1])
print(result)
testset_label_difference()
