import streamlit as st
import pandas as pd
import numpy as np
import random

from utils.prediction import *
from utils.preprocessing import *
from utils.modeling import *

@st.cache_data
def load_dataset():
    return pd.read_csv('data/dataset.csv')

@st.cache_data
def set_labels():
    return [
    "articles for newborns and babies",
    "children's games",
    "children's toys",
    "downloadable video games",
    "figurines",
    "figurines to paint and assemble",
    "food",
    "foreign literature",
    "gaming accessories",
    "garden accessories and decorations",
    "gardening accessories and tools",
    "historical literature",
    "home accessories and decorations",
    "home furnishings and decoration",
    "literature series",
    "model making",
    "non-fiction books",
    "nursery products",
    "outdoor accessories",
    "pet accessories",
    "pool accessories",
    "sets of gaming or video game accessories",
    "stationery",
    "textile accessories and decorations",
    "trading card games",
    "video game consoles",
    "video games"
]

dataset = load_dataset()
labels = set_labels()
im_model = load_image_model()
txt_model = load_text_model()
im_acc, txt_acc = load_accuracy()

def demo():
    if 'id_product' not in globals():
        global id_product
        id_product = 27

    col1, col2 = st.columns([1, 2])
    def prediction_on_button(id_product, mode):
        res = predict_fusion(im_model, txt_model, im_acc, txt_acc, id_product, dataset, labels)
        with col1 :
            with st.expander("TEXT INPUT 📝"):
                st.write(dataset.text.iloc[id_product])
            with st.expander("IMAGE INPUT 🖼️"): 
                st.image(f'data/images/{dataset.image_path.iloc[id_product]}', use_container_width=True)
            with st.expander("RESULT 🎯"):
                if mode == "Texte 📝" or mode == ":rainbow[Fusion] 🌀":
                    if res['true_label'] != res['text_prediction']:
                        st.error(f"📝 : **{res['text_prediction']}**")
                    else : 
                        st.success(f"📝 : **{res['text_prediction']}**")
                if mode == "Image 🖼️" or mode == ":rainbow[Fusion] 🌀":
                    if res['true_label'] != res['image_prediction']:
                        st.error(f"🖼️ : **{res['image_prediction']}**")
                    else : 
                        st.success(f"🖼️ : **{res['image_prediction']}**")
                if mode == ":rainbow[Fusion] 🌀":
                    if res['true_label'] != res['final_prediction']:
                        st.error(f"💡 : **{res['final_prediction']}**")
                    else : 
                        st.success(f"💡 : **{res['final_prediction']}**")
                st.success(f"✅ : **{dataset.prdtypecode.iloc[id_product]}**")
            
        with col2 :
            if mode == ":rainbow[Fusion] 🌀":
                plot_fusion_preds(res, labels, im_acc, txt_acc)

            if mode == "Texte 📝":
                plot_text_prediction(res, labels)

            if mode == "Image 🖼️":
                plot_image_prediction(res, labels)
    
    with col1 :
        col1_1, col1_2 = st.columns([1, 1])
        with col1_1:
            if st.button("Generer un nouvel article 🎲"):
                id_product = random.randint(0, 269)
        with col1_2:
            st.info(f"**article n°{id_product}**")

        mode = st.radio("**Selection du classifier :**",
        ["Texte 📝", "Image 🖼️", ":rainbow[Fusion] 🌀"],
        captions=[
            "Prédiction sur la description de l'article.",
            "Prédiction sur l'image de l'article.",
            "Fusion des prédictions.",
            ],
        index = 0,
        )

    prediction_on_button(id_product, mode)