from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import numpy as np
import pickle
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Set base path for model and data files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

lemmatizer = WordNetLemmatizer()
model = load_model(os.path.join(BASE_DIR, 'chatbot_model.h5'))
words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))

USER = {"username": "user", "password": "pass"}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USER["username"] and password == USER["password"]:
            session["user"] = username
            return redirect(url_for("chat"))
        else:
            flash("Invalid credentials. Try again.")
    return render_template('login.html')

@app.route('/chat')
def chat():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template('index.html', user=session["user"])

@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route('/get_response', methods=['POST'])
def get_response():
    if "user" not in session:
        return jsonify({'response': 'Unauthorized'}), 401

    user_input = request.form['user_input']
    sentence_words = nltk.word_tokenize(user_input)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    prediction = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(prediction) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        intent_index = results[0][0]
        intent = classes[intent_index]
        response = f"Intent: {intent}"
    else:
        response = "Sorry, I didn't understand that."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)