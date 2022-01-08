import pyttsx3
import random
import json
import pickle
import numpy as np
import pywhatkit
import wikipedia
import pyjokes

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('all')


import tensorflow.keras.models


lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tensorflow.keras.models.load_model('chatbotmodel.h5')


text_speech = pyttsx3.init()
voices = text_speech.getProperty('voices')
text_speech.setProperty('voice', voices[1].id)




def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("|============= Welcome to our Smart Chatbot System! =============|")
print("|============================== Feel Free ============================|")
print("|================================== To ===============================|")
print("|=============== Chat with our Intelligent Machine ================|")
while True:
    message = input("| You: ")
    if message == "bye" or message == "Goodbye":
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)
        text_speech.say(res)
        text_speech.runAndWait()
        print("|===================== The Program End here! =====================|")
        exit()
    elif  "about" in message or  "abcd" in message or  "xyz" in message or "123" in message:
        print("| Bot: I did not get it but I am going to search it for you.")
        text_speech.say("I did not get it but I am going to search it for you")
        text_speech.runAndWait()
        pywhatkit.search(message)
    elif 'play' in message:
        print("| Bot: I am going to play it for you.")
        text_speech.say("I am going to play it for you")
        text_speech.runAndWait()
        pywhatkit.playonyt(message)
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)
        text_speech.say(res)
        text_speech.runAndWait()