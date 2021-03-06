# Obligatory assignment 2
# Students: skn003, tre081

import pandas as pd


def classifier(inp):
    # Stores the full vocabulary with total word count
    vocab = pd.read_csv('Vocabulary_small.csv', delimiter = ',') 
    # Stores the full vocabulary with number of times each word occurrs in reliable texts
    vocab_reliable = pd.read_csv('Vocabulary_small_reliable.csv', delimiter = ',') 
    # Stores the full vocabulary with number of times each word occurrs in unreliable texts
    vocab_unreliable = pd.read_csv('Vocabulary_small_unreliable.csv', delimiter = ',')
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
    elif int(avg_reliability) == 0:
        print("This text is reliable!")
        print(avg_reliability)


text_input = input("Enter text: ")
classifier(text_input)
