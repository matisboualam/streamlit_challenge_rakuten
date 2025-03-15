import streamlit as st
import pandas as pd
import random
from models import Model
import subprocess

access_key_id = st.secrets["dvc"]["access_key_id"]

subprocess.run([
    'dvc', 'remote', 'modify', 'origin', '--local', 
    f'access_key_id={access_key_id}'
])

subprocess.run([
    'dvc', 'remote', 'modify', 'origin', '--local', 
    f'secret_access_key={access_key_id}'
])

subprocess.run(['dvc', 'pull'])

st.set_page_config(page_title="Rakuten Multimodal Classifier",
                   page_icon="gallery/rakuten.svg",
                   layout="wide")

@st.cache_data
def load_dataset():
    return pd.read_csv('data/dataset.csv')

dataset = load_dataset()
model = Model()

st.header("IMAGE-TEXT MULTIMODAL CLASSIFIER")
st.divider()

# Fonction pour gérer les prédictions
def prediction_on_button(id_product, mode):
    with st.sidebar:
        st.divider()
        with st.expander("🎯 **RESULTS**", expanded=True):
            if mode == "Texte 📝" or mode == ":rainbow[Fusion] 🌀":
                if st.session_state.res['true_label'] != st.session_state.res['text_prediction']:
                    st.error(f"📝 **{st.session_state.res['text_prediction']}**")
                else: 
                    st.success(f"📝 **{st.session_state.res['text_prediction']}**")
            if mode == "Image 🖼️" or mode == ":rainbow[Fusion] 🌀":
                if st.session_state.res['true_label'] != st.session_state.res['image_prediction']:
                    st.error(f"🖼️ **{st.session_state.res['image_prediction']}**")
                else: 
                    st.success(f"🖼️ **{st.session_state.res['image_prediction']}**")
            if mode == ":rainbow[Fusion] 🌀":
                if st.session_state.res['true_label'] != st.session_state.res['final_prediction']:
                    st.error(f"🌀 **{st.session_state.res['final_prediction']}**")
                else: 
                    st.success(f"🌀 **{st.session_state.res['final_prediction']}**")
            st.info(f"✅ True Label: **{dataset.prdtypecode.iloc[id_product]}**")

    # Affichage des graphiques
    if mode == ":rainbow[Fusion] 🌀":
        model.plot_fusion_preds(st.session_state.res)
    elif mode == "Texte 📝":
        model.plot_text_prediction(st.session_state.res)
    elif mode == "Image 🖼️":
        model.plot_image_prediction(st.session_state.res)

def demo():
    # Sidebar inputs - Image and Text
    with st.sidebar:
        if st.button("🎲 **NEW** 🎲"):
            st.session_state.id_product = random.randint(0, len(dataset) - 1)
            if 'res' not in st.session_state or st.session_state.res.get('id_product') != st.session_state.id_product:
                st.session_state.res = model.predict_fusion(st.session_state.id_product, dataset)
            st.rerun()
        st.divider()
        with st.expander("🖼️ **IMAGE INPUT**"): 
            st.image(f'data/images/{dataset.image_path.iloc[st.session_state.id_product]}', use_container_width=True)
        with st.expander("📜 **TEXT INPUT**"):
            st.write(dataset.text.iloc[st.session_state.id_product])
        
    # Layout pour le bouton de produit aléatoire
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**Article n°{st.session_state.id_product}**")

    with col2:
        # Utiliser st.session_state pour mémoriser la sélection du mode
        mode = st.radio(
            "🎛 **Select Classifier:**",
            ["Texte 📝", "Image 🖼️", ":rainbow[Fusion] 🌀"],
            captions=[
                "Prediction based on text description.",
                "Prediction based on image.",
                "Fusion of both predictions.",
            ],
            index=["Texte 📝", "Image 🖼️", ":rainbow[Fusion] 🌀"].index(st.session_state.get('mode', ":rainbow[Fusion] 🌀")),  # Récupérer le mode actuel
            horizontal=True
        )
        
        # Mettre à jour le mode dans st.session_state
        st.session_state.mode = mode

    # Mettre à jour les prédictions et afficher les résultats
    prediction_on_button(st.session_state.id_product, st.session_state.mode)

    st.divider()
    with st.expander("**Dataset Content**"):
        st.dataframe(dataset, use_container_width=True, height=200)

# Initialiser les variables dans st.session_state si elles ne sont pas déjà présentes
if 'id_product' not in st.session_state:
    st.session_state.id_product = random.randint(0, len(dataset) - 1)

if 'res' not in st.session_state or st.session_state.res.get('id_product') != st.session_state.id_product:
    st.session_state.res = model.predict_fusion(st.session_state.id_product, dataset)

if 'mode' not in st.session_state:
    st.session_state.mode = "Texte 📝"  # Valeur par défaut

demo()
