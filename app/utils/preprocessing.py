import numpy as np
import tensorflow as tf
import re
import string
import nltk
import os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from utils.nltk_setup import setup_nltk
from utils.modeling import load_word2vec_model


word2vec_model = load_word2vec_model()
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

def remove_isolated_letter_apostrophe(text):
    return re.sub(r"\b[a-zA-Z]'", "", text)

def get_mean_vector(model, text):
    if model is None:
        return np.zeros((model.vector_size,))  # ou toute autre taille de vecteur appropriée
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:  # Vérifiez si le vecteur des mots est vide
        return np.zeros((model.vector_size,))  # ou une autre valeur par défaut
    return np.mean(word_vectors, axis=0)

def remove_isolated_letter_apostrophe(text):
    return re.sub(r"\b[a-zA-Z]'", "", text)

def preprocess_text(text):
    text = text.lower()
    text = remove_isolated_letter_apostrophe(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    return text.lower()

def preprocess_text_data(text_data):
    text_input = preprocess_text(text_data)
    text_input = clean_text(text_input)
    text_input = np.array([get_mean_vector(word2vec_model, text_input)])
    return text_input

def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
