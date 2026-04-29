import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import cleaning

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('spam_model.keras',compile=False)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl","rb") as f:
        return pickle.load(f)
    
model=load_model()
tokenizer=load_tokenizer()

maxulength=100

def predict_spam(text):
    cleaned=cleaning(text)
    sequence=tokenizer.texts_to_sequences([cleaned])
    padded=pad_sequences(sequence,maxlen=maxulength,padding='post',truncating='post')
    prediction=model.predict(padded)[0][0]
    return prediction

st.title("SPAM EMAIL CLASSIFIER")
email_input=st.text_area("Enter email text")
if st.button('PREDICT'):
    if email_input.strip()=="":
        st.warning('Please Enter Some Text!')
    else:
        probability=predict_spam(email_input)
        if probability>0.5:
            st.error(f"Spam Detected!!!(Confidence:{probability*100:.1f}%)")
        else:
            st.success(f"NOt SPAM :)({(1-probability)*100:.1f}%)")