import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import re
import string
from bs4 import BeautifulSoup
import nltk
import gensim
import json
import streamlit as st
import matplotlib.pyplot as plt

word2vec_model = gensim.models.KeyedVectors.load('models/gensim/fasttext-wiki-news-subwords-300')
stop_words = set(nltk.corpus.stopwords.words('french'))
lemmatizer = nltk.stem.WordNetLemmatizer()

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
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    return text.lower()


# Cache Word2Vec Model
@st.cache_data
def load_word2vec_model():
    return gensim.models.KeyedVectors.load('models/gensim/fasttext-wiki-news-subwords-300')

# Cache stop words
@st.cache_data
def get_stop_words():
    return set(nltk.corpus.stopwords.words('french'))

# Cache lemmatizer
@st.cache_data
def get_lemmatizer():
    return nltk.stem.WordNetLemmatizer()

# Cache Keras models
@st.cache_data
def load_text_model():
    return load_model('models/text_model.keras')

@st.cache_data
def load_image_model():
    return load_model('models/image_model_MobileNet.keras')

# Cache Accuracy DataFrames
@st.cache_data
def load_im_accuracy():
    return pd.read_csv('data/accuracy/im_accuracy_MobileNet.csv', index_col=0)

@st.cache_data
def load_txt_accuracy():
    return pd.read_csv('data/accuracy/txt_accuracy.csv', index_col=0)

# Cache catalog
@st.cache_data
def load_catalog():
    return json.load(open('models/catalog.json'))


class Model:
    def __init__(self):
        # Load models and resources
        self.word2vec_model = load_word2vec_model()
        self.stop_words = get_stop_words()
        self.lemmatizer = get_lemmatizer()
        self.text_model = load_text_model()
        self.image_model = load_image_model()
        self.im_acc = load_im_accuracy()
        self.txt_acc = load_txt_accuracy()
        self.catalog = load_catalog()

        # Delete heavy files after they are loaded into memory
        self.cleanup()

    def cleanup(self):
        # Optionally delete the Word2Vec model file after loading
        word2vec_model_path = 'models/gensim/fasttext-wiki-news-subwords-300'
        
        if os.path.exists(word2vec_model_path):
            os.remove(word2vec_model_path)
            print(f"Word2Vec model file '{word2vec_model_path}' deleted from disk.")

    def preprocess_text_data(self, text_data):
        text_input = preprocess_text(text_data)
        text_input = clean_text(text_input)
        text_input = np.array([get_mean_vector(self.word2vec_model, text_input)])
        return text_input

    def preprocess_image(self, image_path, target_size=(224, 224)):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict_text(self, text):
        text_input = self.preprocess_text_data(text)
        pred_text = self.text_model.predict(text_input, verbose=0)
        return pred_text
    
    def predict_image(self, image_path):
        image_input = self.preprocess_image(image_path)
        pred_im = self.image_model.predict(image_input, verbose=0)
        return pred_im
    
    def predict_fusion(self, article, dataset):
        pred_txt = self.predict_text(dataset.text.iloc[article])
        pred_im = self.predict_image(f'data/images/{dataset.image_path.iloc[article]}')

        predicted_class_index_im = np.argmax(pred_im[0])
        predicted_class_index_txt = np.argmax(pred_txt[0])

        weighted_pred_im = pred_im[0] * self.im_acc.values.flatten()
        weighted_pred_txt = pred_txt[0] * self.txt_acc.values.flatten()

        weighted_pred_im /= np.sum(weighted_pred_im)
        weighted_pred_txt /= np.sum(weighted_pred_txt)

        combined_probs = (weighted_pred_im + weighted_pred_txt) / 2

        final_class_index = np.argmax(combined_probs)

        return {
            'text_input': dataset.text.iloc[article],
            'image_input': dataset.image_path.iloc[article],
            'pred_im': pred_im,
            'pred_txt': pred_txt,
            'weighted_pred_im': weighted_pred_im,
            'weighted_pred_txt': weighted_pred_txt,
            'combined_probs': combined_probs,
            'image_prediction': self.catalog[predicted_class_index_im],
            'text_prediction': self.catalog[predicted_class_index_txt],
            'final_prediction': self.catalog[final_class_index],
            'true_label': dataset.prdtypecode.iloc[article]
        }

    # Add plotting methods here as before...
