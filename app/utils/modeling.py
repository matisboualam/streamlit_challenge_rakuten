import gensim
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_word2vec_model():
    return gensim.models.KeyedVectors.load('models/gensim/fasttext-wiki-news-subwords-300')

def load_image_model():
    return tf.keras.models.load_model('models/image_model_MobileNet.keras')

def load_text_model():
    return tf.keras.models.load_model('models/text_model.keras')

def load_accuracy():
    im_acc = pd.read_csv('data/accuracy/im_accuracy_MobileNet.csv', index_col=0)
    txt_acc = pd.read_csv('data/accuracy/txt_accuracy.csv', index_col=0)
    return im_acc, txt_acc


def plot_text_prediction(res, class_labels):
    x = range(len(class_labels))
    true_label_index = class_labels.index(res['true_label'])

    fig, ax = plt.subplots(figsize=(12, 6)) 
    ax.set_title('Text Model Weighted Prediction')
    ax.plot(x, res['weighted_pred_txt'], color='salmon', marker='o', linestyle='-', label='Text Model Weighted Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability')
    if res['true_label'] == res['text_prediction']:
        ax.axvline(x=true_label_index, color='green', linestyle='--', label='Final Prediction')
    else:
        ax.axvline(x=true_label_index, color='green', linestyle='--', label='True Label')
        ax.axvline(x=class_labels.index(res['text_prediction']), color='red', linestyle='--', label='Final Prediction')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)

def plot_image_prediction(res, class_labels):
    x = range(len(class_labels))
    true_label_index = class_labels.index(res['true_label'])
    fig, ax = plt.subplots(figsize=(12, 6))  # Default size, but this will adjust with `use_container_width`
    ax.set_title('Image Model Weighted Prediction')
    ax.plot(x, res['weighted_pred_im'], color='skyblue', marker='o', linestyle='-', label='Image Model Weighted Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability')

    if res['true_label'] == res['image_prediction']:
        ax.axvline(x=true_label_index, color='green', linestyle='--', label='Final Prediction')
    else:
        ax.axvline(x=true_label_index, color='green', linestyle='--', label='True Label')
        ax.axvline(x=class_labels.index(res['image_prediction']), color='red', linestyle='--', label='Final Prediction')

    ax.legend()
    ax.grid(True)
    st.pyplot(fig, use_container_width=True) 
    
def plot_fusion_preds(res, class_labels, im_acc, txt_acc):
    x = range(len(class_labels))
    true_label_index = class_labels.index(res['true_label'])

    fig, ax = plt.subplots(figsize=(12, 6)) 
    ax.plot(x, res['combined_probs'], color='purple', marker='o', linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_title('Combined Predictions')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability')
    ax.axvline(x=true_label_index, color='green', linestyle='--', label='True Label')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)  # Display the plot in Streamlit with container width

    fig, ax = plt.subplots(figsize=(12, 6))  # Default size, but this will adjust with `use_container_width`
    ax.set_title('Weighted Predictions and Accuracy for Image and Text')
    ax.plot(x, res['weighted_pred_im'], color='skyblue', marker='o', linestyle='-', label='Image Model Weighted Prediction')
    ax.plot(x, res['weighted_pred_txt'], color='salmon', marker='o', linestyle='-', label='Text Model Weighted Prediction')
    ax.fill_between(x, 0, im_acc['precision']/3, color='blue', alpha=0.2, label='Image Model Accuracy')
    ax.fill_between(x, 0, txt_acc['precision']/3, color='red', alpha=0.2, label='Text Model Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability / Accuracy')
    ax.legend()
    ax.grid(True)
    with st.expander("VISUALISATION DE LA PONDERATION"):
        st.pyplot(fig, use_container_width=True) 