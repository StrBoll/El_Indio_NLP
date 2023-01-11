from nltk.translate.ribes_score import sentence_ribes
import random
import json
import pickle
import numpy
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import streamlit as st
from streamlit_chat import message

lemmatizer = WordNetLemmatizer()
dictionary = json.loads(open("dictionary.json").read())
CleanedWords = pickle.load(open('cleanedwords.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('IndioNLP.h5')


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


def generate_lemma(UserInput):
    remove_punctuation = ['?', '!', '.', ')', '(', '%', '>', '<', '#', '^', '-', '&']
    cleaned_words = []
    cleaned_sentence = []
    initial_sentence_phase = nltk.pos_tag(nltk.word_tokenize(UserInput))
    lemmatized_sentence = list(map(lambda x: (x[0], tagger(x[1])), initial_sentence_phase))
    for word, type_tag in lemmatized_sentence:
        if type_tag is None:
            cleaned_words.append(word)
            cleaned_sentence.append(word)
        else:
            cleaned_words.append(lemmatizer.lemmatize(word, type_tag))
            cleaned_sentence.append(lemmatizer.lemmatize(word, type_tag))
    cleaned_words = [word.lower() for word in cleaned_words]
    for word in cleaned_words:
        if word in remove_punctuation:
            cleaned_words.remove(word)



    return cleaned_words


def bag(UserInput):
    sentence = generate_lemma(UserInput)
    bag = [0 for word in range(len(CleanedWords))]
    for wordX in sentence:
        for index, wordY in enumerate(CleanedWords):
            if wordY == wordX:
                bag[index] = 1
    return numpy.array(bag)


def Conversate():
    print("Hello, welcome to the El Indio Chat AI. Type quit to end the conversation.")

    while True:
        Sentence_Input = input("Me: ")

        if Sentence_Input.lower() == 'quit':
            break
        bow = bag(Sentence_Input)
        results = model.predict(numpy.array([bow]))[0]
        return_list = numpy.argmax(results)

        cat = classes[return_list]

        for tag in dictionary['intents']:
            if tag['tag'] == cat:
                responses = tag['responses']
                response = random.choice(responses)
                print(f'Guy: {response}')




if __name__ == '__main__':

    Conversate()


