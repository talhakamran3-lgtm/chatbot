import random
import json
import pickle
import numpy as np
import nltk
import tkinter as tk
from tkinter import scrolledtext, END

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- Your Existing Chatbot Logic (unchanged) ---
lemmatizer = WordNetLemmatizer()
# Note: Ensure these files are in the same directory as the script, or provide the full path.
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
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
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I don't understand that. Could you please rephrase?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:
        result = "I'm not sure how to respond to that."
    return result

# --- New GUI Code ---

def send_message(event=None):
    """
    Handles sending the user's message and displaying the bot's response.
    """
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if msg != '':
        # Display user's message
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))

        # Get and display bot's response
        ints = predict_class(msg)
        res = get_response(ints, intents)
        chat_log.insert(tk.END, "Bot: " + res + '\n\n')

        # Make the chat log uneditable and scroll to the end
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)

# --- Setting up the main window ---
root = tk.Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)

# --- Creating the GUI widgets ---

# Create the chat window (log)
chat_log = scrolledtext.ScrolledText(root, bd=0, bg="white", height="8", width="50", font="Arial")
chat_log.config(state=tk.DISABLED)  # Make it read-only

# Create the Send button
send_button = tk.Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                        command=send_message)

# Create the message entry box
entry_box = tk.Text(root, bd=0, bg="white", width="29", height="5", font="Arial")

# Bind the <Return> key to the send_message function
root.bind('<Return>', send_message)


# --- Placing the widgets on the screen ---
chat_log.place(x=6, y=6, height=386, width=386)
entry_box.place(x=6, y=401, height=90, width=265)
send_button.place(x=277, y=401, height=90)

# --- Start the GUI event loop ---
print("GO! Bot is running!")
root.mainloop()