import pandas as pd
import numpy as np

data=pd.read_csv('languages_file.csv')

data

data['language'].value_counts()

print(data[data['language']=='Hindi'])

#checking for null value and solving taht
data.isnull().sum()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Conversion of text and language column from pandas to array
Text_to_array=np.array(data['Text'])
Language_to_array=np.array(data['language'])

print(Text_to_array)
print(Language_to_array)

Text=Text_to_array
Language=Language_to_array

#Converting the text arary to numbers [step before feedign the values to machine]
TN=CountVectorizer()
Text=TN.fit_transform(Text)

Text_train,Text_test,Language_train,Language_test=train_test_split(Text,Language,test_size=0.2, random_state=22)

print(Text)

#start building model
model=MultinomialNB()
model.fit(Text_train,Language_train)

print(model.score(Text_test,Language_test))

User=input("Enter a Text: ")
data=TN.transform([User]).toarray()
output=model.predict(data)
print(output)

import streamlit as st
st.set_page_config(page_title="Language detection", layout='centered')
st.title("Detecting language")

user_input=st.text_area("Enter a sentence")

if st.button("Detect language"):
    if user_input.strip()=="":
        print("Eneter some text")
    else:
        try:
            input_vector=vector.transform([user_input])
            output=model.predict(input_vector)
            st.success(f"Predicted language : ** {output[0]}**")
        except Exception as e:
            st.error(f"Error : {e} ")




