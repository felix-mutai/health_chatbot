import streamlit as st
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import load_model
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

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

    return "Hmm, I couldn't find a good answer to that."

# Streamlit UI setup
st.set_page_config(page_title="Health Chatbot", page_icon="💬")
st.title("🩺 Health Chatbot")
st.markdown("Ask a health-related question and I’ll try to help!")

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input with a dedicated key
user_input = st.text_input("👤 You:", key="user_input")

# Process input and update chat
if user_input.strip():  # Check for non-empty input
    predictions = predict_class(user_input)
    bot_response = get_response(predictions, intents)
    
    # Store messages
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))
    
    # Clear input field after processing
    st.session_state.user_input = ""

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"👤 **You**: {message}")
    else:
        st.markdown(f"💬 **Bot**: {message}")

st.markdown("---")
st.markdown("You can continue asking more health-related questions or type 'exit' to stop.")
