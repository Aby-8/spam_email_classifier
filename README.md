
# Spam Email Classifier (LSTM + Streamlit)

This project detects whether an email is spam or not using a deep learning LSTM model and provides predictions through a Streamlit web application.

## Features

* Text preprocessing pipeline (lowercasing, punctuation removal, stopword handling)
* Tokenization and sequence padding
* LSTM neural network classifier
* Probability-based prediction output
* Interactive Streamlit interface

## Tech Stack

* Python
* TensorFlow / Keras
* NLTK
* Streamlit

## Model Architecture

Embedding → LSTM → Dense → Sigmoid Output

## Dataset

Email spam dataset with labeled spam and ham messages used for supervised training.

## How to Run Locally

1. Clone this repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

## Project Structure

* app.py – Streamlit interface
* preprocessing.py – text cleaning pipeline
* spam_model.keras – trained model
* tokenizer.pkl – tokenizer mapping

## Future Improvements

* Bidirectional LSTM upgrade
* Transformer-based classifier
* REST API deployment using FastAPI
