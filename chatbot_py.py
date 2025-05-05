import streamlit as st
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import load_model

# Initialize tokenizer and lemmatizer
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

# Load resources
with open('intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Clean and preprocess input
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that."

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    # If no intent matched
    return "Hmm, I couldn't find a good answer to that."

# Streamlit UI
st.set_page_config(page_title="Health Chatbot", page_icon="💬")
st.title("🩺 Health Chatbot")
st.markdown("Ask a health-related question and I’ll try to help!")

user_input = st.text_input("👤 You:", "")

if user_input:
    predictions = predict_class(user_input)
    response = get_response(predictions, intents)
    st.markdown(f"💬 **Bot:** {response}")
