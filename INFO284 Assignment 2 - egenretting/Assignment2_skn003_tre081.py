# Obligatory assignment 2
# Students: skn003, tre081
from collections import defaultdict

import numpy as np
import pandas as pd

# Assignment 2 prep

# Reading vocabulary and dataset for future use
vocabulary = pd.read_csv('Vocabulary.csv', delimiter=',')
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
        self.initial_log_probability = defaultdict(defaultdict)
        self.log_probabilities = defaultdict(dict)

    # Predict
    # test is the text or .txt file that will be evaluated
    def predict(self, test):
        text = ''
        # If the input is a filename we open and read the associated file
        if str(test)[-4:] == '.txt':
            file = open(test, 'r')
            text = file.read()
            file.close()
        else:
            text = test
        sums = {0: 0, 1: 0}
        for label in [0, 1]:
            sums[label] = self.initial_log_probability[label]
            words = str(text).split(" ")
            for w in words:
                if w in self.vocab_words:
                    # Adds the log for a given word with the current label if that word is in the vocabulary
                    sums[label] += self.log_probabilities[label][w]
        return sums

    # train
    # Calculates the log likelihoods for all the words in the vocabulary based on label
    def train(self):
        alpha = 1
        word_counts_by_label = {}
        total_texts = len(train['text'])

        for t, l in zip(train['text'], train['label']):
            self.text_label[l].append(t)

        # Counting how many times each word in the vocabulary is in texts with a given label
        for k in [0, 1]:
            all_texts = self.text_label[k]
            word_counts_by_label[k] = defaultdict(int)
            for t in all_texts:
                words = str(t).split(" ")
                for w in words:
                    word_counts_by_label[k][w] += 1

        # Doing the calculations
        for label in [0, 1]:
            label_total_count = sum(train['label'] == label)
            # Calculating the initial probability for the current label
            self.initial_log_probability[label] = np.log(label_total_count / total_texts)
            total_count = 0
            for w in self.vocab_words:
                # Adding up how many words appear and how often they appear in texts with the current label
                total_count += word_counts_by_label[label][w]

            for w in self.vocab_words:
                c = word_counts_by_label[label][w]
                # Calculating the probability for each word
                self.log_probabilities[label][w] = np.log((c + alpha) / (total_count + alpha * len(self.vocab)))


'''
__Function__ 
testset_label_difference:
    When the classifier is run
    Increment incorrect_label by 1 for each incorrectly labelled text
'''
def testset_label_difference():
    c = Classifier()
    c.train()
    testset = train['text']
    target = train['label']
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

    for n in range(testset_length):
        predict_result = c.predict(testset[n])
        new_label = 0
        if predict_result[0] < predict_result[1]:
            new_label = 1  # Running the classifier on text in test set
        if target[n] != new_label:
            incorrect_label += 1  # Increase the count if the classifier is wrong

    error_rate_calculator()  # Calculating the results


classifier = Classifier()
classifier.train()
inp = input("Please enter text or .txt file path > ")
result = classifier.predict(inp)
print("Prediction: ", result)
testset_label_difference()
