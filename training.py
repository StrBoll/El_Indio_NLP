import json
import pickle
import random
import nltk
import numpy

nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('omw-1.4')
import tensorflow
from tensorflow import keras
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# I created a number of empty lists to store the patterns, lemmatized_words, and tags of the dictionary.json file

cleaned_words = []
# cleaned_words holds the lemma which will be converted into binary for training the neural network
classes = []
# classes will hold the tags that represent groups of patterns and responses
categories = []
# category will hold the patterns in conjunction to their respective tags

remove_punctuation = ['?', '!', "'", ',', '.', '-']


# this library will represent that characters that must be removed from the lemma before being processed into binary

# tagger function will append the POS tag to each lemmatized word
def tagger(tag):
    if tag[0] == 'J':
        return wordnet.ADJ
    elif tag[0] == 'R':
        return wordnet.ADV
    elif tag[0] == 'V':
        return wordnet.VERB
    elif tag[0] == 'N':
        return wordnet.NOUN
    else:
        return


# Next, I create a variable called dictionary which stores the intents, patterns and responses of the intents library

with open('dictionary.json') as intents:
    dictionary = json.load(intents)

# Now, I iterate through the dictionary.json file to extract and lemmatize patterns

for intent in dictionary['intents']:
    for pattern in intent['patterns']:

        pos_complete = nltk.pos_tag(nltk.word_tokenize(pattern))

        # Now to simplify things, we use the pos tag function we created,
        finished_tag = list(map(lambda x: (x[0], tagger(x[1])), pos_complete))

        cleaned_sentence = []
        # initializing empty list through each iteration to hold the full lemmatized sentence that will be appended to
        # class variable

        for word, type_tag in finished_tag:

            if type_tag is None:

                # this means there isn't any tags available to append, so we append it as-is
                cleaned_words.append(word)
                cleaned_sentence.append(word)

            else:

                cleaned_words.append(lemmatizer.lemmatize(word, type_tag))
                cleaned_sentence.append(lemmatizer.lemmatize(word, type_tag))

        # append the pattern and tag to category variable
        categories.append((cleaned_sentence, intent['tag']))

        classes.append(intent['tag'])
        # append the tags from dictionary.json into the classes variable, we will later remove duplicates and sort it

# Precautionary measure to ensure all lemma are completely lowercase
cleaned_words = [word.lower() for word in cleaned_words]

# any punctuation will be removed from the lemma list

for word in cleaned_words:
    if word in remove_punctuation:
        cleaned_words.remove(word)
# we will sort the list to remove any duplicates, then sort the list lexicographically
cleaned_words = sorted(set(cleaned_words))

# sort and remove duplicates from class
classes = sorted(set(classes))

# Now that the lemmatization process is over, it's time to save the lemma and classes to binary files using pickle

pickle.dump(cleaned_words, open('cleanedwords.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create variable called numerical_rep which will be used to numerically classify and store the data into 1s and 0s
# This classification is to allow the neural network I'm training to understand the data since it cannot work
# with strings

numerical_rep = []

# output_empty is a common identifier in python, in our case we will be using it to store a singular 0 for every class
# in the classes variable
output_empty = [0] * len(classes)

# in the next for-loop, I will append 1 to the bag variable if the word being examined matches the word in the current
# pattern and 0 if it does not match
# Afterwards we will use the numpy library to create a numpy array out of these 0s and 1s which will then be shuffled
# for training

for category in categories:
    pattern_words = category[0]
    bag = []
    for word in pattern_words:
        pattern_words[pattern_words.index(word)] = lemmatizer.lemmatize(word.lower())
    for lemma in cleaned_words:
        if lemma in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # here I made a shallow copy of output_empty
    output_row = list(output_empty)
    output_row[classes.index(category[1])] = 1

    # Now I append the bag of numerically represented word-relations and the output row
    numerical_rep.append([bag, output_row])

random.shuffle(numerical_rep)
# Shuffle the numerical_rep training data and make a numpy array

training_data = numpy.array(numerical_rep)

# Now, I split the data into separate groups, the first group contains the 1s and the second group contains the 0s

y_data = list(training_data[:, 0])
x_data = list(training_data[:, 1])

# There are a ton of different ways to train neural networks for natural language processing
# However, in the case of this dataset, we have organized the data into a plain stack of layers
# For this model, I will use a sequential model from the keras subpackage of tensorflow
# https://keras.io/guides/sequential_model/

# I created this sequential model via a keras sequential constructor

# Dropout function is used to avoid over training the model, this would lead to memorization of the data set and
# decrease the model's ability to predict using the data


model = Sequential(
    [
        Dense(128, input_shape=(len(y_data[0]), ), activation="relu"),
        # I train the model with the input shape being the length of each data set that way it automatically updates
        # the training as the dictionary.json file is expanded
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(len(x_data[0]), activation="softmax")

    ]
)

# https://keras.io/api/layers/activations/
# relu token refers to rectified linear unit activation function
# softmax token refers to keras function which converts vectors of values into a probability distribution

# Now to compile the model using Gradient Descent
# SGD automatically sets learning_rate to 0.01, momentum to 0.0 and nesterov to False
# https://keras.io/api/optimizers/sgd/
# However, we're going to change a few of these parameters to fit the HUGE dataset

# Here, I'm using a constant learning rate of 0.01 as opposed to other methods which are usually more complex
# Here's an article on learning rate schedules I found useful on determining mine
# https://towardsdatascience.com/learning-rate-schedule-in-practice-an-example-with-keras-and-tensorflow-2-0-2f48b2888a0c
optimizer = SGD(learning_rate=0.01, nesterov=True, decay=1e-6, momentum=0.8)

# After fiddling with momentum, batch size, and epochs, I was able to keep the accuracy around 93%
# and the loss around 20%
# Essentially, this means that the model will predict correctly 90% of the time,
# however in the 10% chance it predicts incorrectly
# the error is only 20% off

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(numpy.array(y_data), numpy.array(x_data), epochs=300, batch_size=7, verbose=1)
model.save("IndioNLP.h5", history)

# Epochs set to 300 with a batch size of 7 and momentum tweaked to 0.8, the model's highest rating was an accuracy of
# 95.88% with a loss of 14.27%