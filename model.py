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
import subprocess

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

class Model:
    def __init__(self):
        self.text_model = load_model('models/text_model.keras')
        self.image_model = load_model('models/image_model_MobileNet.keras')
        self.im_acc = pd.read_csv('data/accuracy/im_accuracy_MobileNet.csv', index_col=0)
        self.txt_acc = pd.read_csv('data/accuracy/txt_accuracy.csv', index_col=0)
        self.catalog = json.load(open('models/catalog.json'))
    
    def preprocess_text_data(self, text_data):
        text_input = preprocess_text(text_data)
        text_input = clean_text(text_input)
        text_input = np.array([get_mean_vector(word2vec_model, text_input)])
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
     
    def plot_text_prediction(self, res):
        col1, col2 = st.columns([1,1])
        x = range(len(self.catalog))
        true_label_index = self.catalog.index(res['true_label'])

        fig, ax = plt.subplots(figsize=(12, 6)) 
        ax.set_title('Text Model Weighted Prediction')
        ax.plot(x, res['weighted_pred_txt'], color='salmon', marker='o', linestyle='-', label='Text Model Weighted Prediction')
        ax.set_xticks(x)
        ax.set_xticklabels(self.catalog, rotation=90)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Probability')
        if res['true_label'] == res['text_prediction']:
            ax.axvline(x=true_label_index, color='green', linestyle='--', label='Final Prediction')
        else:
            ax.axvline(x=true_label_index, color='green', linestyle='--', label='True Label')
            ax.axvline(x=self.catalog.index(res['text_prediction']), color='red', linestyle='--', label='Final Prediction')
        ax.legend()
        ax.grid(True)
        with col1:
            with st.expander("📊 **Probability Per Class**", expanded=True):
                st.pyplot(fig, use_container_width=True)

    def plot_image_prediction(self, res):
        col1, col2 = st.columns([1,1])
        x = range(len(self.catalog))
        true_label_index = self.catalog.index(res['true_label'])
        fig, ax = plt.subplots(figsize=(12, 6))  # Default size, but this will adjust with `use_container_width`
        ax.set_title('Image Model Weighted Prediction')
        ax.plot(x, res['weighted_pred_im'], color='skyblue', marker='o', linestyle='-', label='Image Model Weighted Prediction')
        ax.set_xticks(x)
        ax.set_xticklabels(self.catalog, rotation=90)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Probability')

        if res['true_label'] == res['image_prediction']:
            ax.axvline(x=true_label_index, color='green', linestyle='--', label='Final Prediction')
        else:
            ax.axvline(x=true_label_index, color='green', linestyle='--', label='True Label')
            ax.axvline(x=self.catalog.index(res['image_prediction']), color='red', linestyle='--', label='Final Prediction')

        ax.legend()
        ax.grid(True)
        with col1:
            with st.expander("📊 **Probability Per Class**", expanded=True):
                st.pyplot(fig, use_container_width=True) 
        
    def plot_fusion_preds(self, res):
        col1, col2 = st.columns([1,1])
        x = range(len(self.catalog))
        true_label_index = self.catalog.index(res['true_label'])

        fig, ax = plt.subplots(figsize=(12, 6)) 
        ax.plot(x, res['combined_probs'], color='purple', marker='o', linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels(self.catalog, rotation=90)
        ax.set_title('Combined Predictions')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Probability')
        ax.axvline(x=true_label_index, color='green', linestyle='--', label='True Label')
        ax.axvline(x=self.catalog.index(res['final_prediction']), color='red', linestyle='--', label='Final Prediction')
        ax.legend()
        ax.grid(True)
        with col1:
            with st.expander("📊 **Probability Per Class**", expanded=True):
                st.pyplot(fig, use_container_width=True)  # Display the plot in Streamlit with container width

        fig, ax = plt.subplots(figsize=(12, 6))  # Default size, but this will adjust with `use_container_width`
        ax.set_title('Weighted Predictions and Accuracy for Image and Text')
        ax.plot(x, res['weighted_pred_im'], color='skyblue', marker='o', linestyle='-', label='Image Model Weighted Prediction')
        ax.plot(x, res['weighted_pred_txt'], color='salmon', marker='o', linestyle='-', label='Text Model Weighted Prediction')
        ax.fill_between(x, 0, self.im_acc['precision']/3, color='blue', alpha=0.2, label='Image Model Accuracy')
        ax.fill_between(x, 0, self.txt_acc['precision']/3, color='red', alpha=0.2, label='Text Model Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(self.catalog, rotation=90)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Probability / Accuracy')
        ax.legend()
        ax.grid(True)
        with col2:
            with st.expander("📊 **Ponderated Sum Vizualization**", expanded=True):
                st.pyplot(fig, use_container_width=True) 
        
