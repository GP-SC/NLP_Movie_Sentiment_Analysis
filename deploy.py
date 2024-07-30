import pandas as pd
import numpy as np
import os
import nltk
from nltk.stem import PorterStemmer
import string
from collections import Counter
import matplotlib.pyplot as plt
nltk.download('punkt')  # Tokenizer data
nltk.download('averaged_perceptron_tagger')  # POS tagging data
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.metrics import classification_report,confusion_matrix
import joblib
import streamlit as st

def remove_punc(text):
    words = nltk.word_tokenize(text)
    stemsent=""
    all_marks =["«", "»","‘", "’", "‚", "‛","“", "”", "„", "‟","‹", "›",
    "❛", "❜", "❝", "❞","``", "〞", "〟","＂", "＇","′", "″", "‴", "⁗","‵", "‶", "‷"
    ]+list(string.punctuation)
    for word in words:
        if word not in all_marks:
            stemsent=stemsent+" "+word
    return stemsent.strip()
def stem(sent):
# Initialize the PorterStemmer
    stemmer = PorterStemmer()
    # Example text
    text = sent
    # Tokenize the text
    words = nltk.word_tokenize(text)
    stemsent=""
    for word in words:
        if word not in string.punctuation:
            stemsent=stemsent+" "+word
    # Stem each word in the text
    
    return stemsent.strip()
def descriptors_extractor(text):
    # Tokenize the text into words

    words = word_tokenize(text)

    # Perform part-of-speech tagging
    pos_tags = pos_tag(words)

    sent=""
    for word, tag in pos_tags:
        if tag.startswith('JJ') or  tag.startswith('RB') or tag.startswith('VB'):
            sent=sent+" "+word
    return sent.strip()
tfv=joblib.load("tfv.pickle")
model=joblib.load("model.pickle")
def predict(txt):
    txt=remove_punc(txt)
    txt=descriptors_extractor(txt)
    txt=stem(txt)
    X=tfv.transform([txt])
    pred=model.predict(X)
    if(pred==1):
        return "This is Positive Review 👍"
    else:
        return "This is Negative Review 👎"
# Streamlit app
def main():
    st.title("🎬Movie Sentiment Analysis")
    st.write("Enter a movie review text to analyze its sentiment.")

    # Input text box for user to enter review
    user_input = st.text_area("Enter your movie review here:")
    
    # Button to trigger sentiment analysis
    if st.button("Analyze"):
        if user_input.strip() != "":
            sentiment = predict(user_input)
            st.write("Sentiment:", sentiment)
        else:
            st.warning("Please enter a movie review.")

# Run the app
if __name__ == "__main__":
    main()