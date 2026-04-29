import string
from nltk.corpus import stopwords
import streamlit as st

@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))

stop_words=load_stopwords()

def cleaning(text):
    text=text.replace("Subject","")
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    words=text.split()
    words=[word for word in words if word not in stop_words]
    return " ".join(words)