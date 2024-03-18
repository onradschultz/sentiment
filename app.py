
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('wordnet')

# Load the trained model
model = joblib.load('finalized_model.pkl')

# Function to clean and lemmatize text
def clean_and_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return lemmatized_text

# Streamlit app
st.title('Sentiment Analysis')

input_text = st.text_area("Enter text to analyze:")

if st.button('Analyze'):
    cleaned_text = clean_and_lemmatize(input_text)
    prediction = model.predict([cleaned_text])[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    st.write(f'Sentiment: {sentiment}')
