# from flask import Flask
# from app import app
# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

# import files
from urllib import response
from flask import Flask, render_template, request

import json 
import numpy as np
from tensorflow import keras

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import os


import random
import pickle


app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def chat():
    # response = request.args.get('msg')
    input_text = request.args.get('msg')
    cwd = os.getcwd()
    path = os.path.join(os.getcwd(), 'intents.json')
    token_path = os.path.join(os.getcwd(), 'tokenizer.pickle')
    label_path = os.path.join(os.getcwd(), 'label_encoder.pickle')
    model_path = os.path.join(os.getcwd(), 'chat_model')

    with open(path) as file:
        data = json.load(file)
    # load trained model
    model = keras.models.load_model(model_path)

    # load tokenizer object
    with open(token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open(label_path, 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    # while True: Jika di command prompt harus looping 
    inp = input_text
    # if inp.lower() == "quit":
    #     break

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                            truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
            response = np.random.choice(i['responses'])
            

    return str(response)

if __name__ == "__main__":
    app.run()
    app.run(APP_DEBUG=False, host="0.0.0.0")
