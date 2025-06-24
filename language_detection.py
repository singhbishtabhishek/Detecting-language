import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load data
data = pd.read_csv('languages_file.csv')

# Convert text and label to arrays
texts = np.array(data['Text'])
labels = np.array(data['language'])

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=22)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Language Detection", layout='centered')
st.title("🌐 Detect Language from Text")

user_input = st.text_area("Enter a sentence:")

if st.button("Detect language"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)
            st.success(f" Predicted language: **{prediction[0]}**")
        except Exception as e:
            st.error(f" Error: {e}")
