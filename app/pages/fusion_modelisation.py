import streamlit as st
import pandas as pd

@st.cache_data
def load_accuracy():
    im_acc = pd.read_csv('/app/data/accuracy/im_accuracy_MobileNet.csv', index_col=0)
    txt_acc = pd.read_csv('/app/data/accuracy/txt_accuracy.csv', index_col=0)
    return im_acc, txt_acc

def fusion_model():
    st.title("Modélisation - fusion multimodale")
    im_acc, txt_acc = load_accuracy()
    with st.expander("PREDICTION PAR PONDERATION"):
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("""
                        - **Fusion Multimodale :** Prédiction par combinaison des modèles de classification par image et par texte :
                        """)
            st.image('/app/gallery/pred_formula.png')
            
        with col2:
            col2_1, col2_2 = st.columns([1,1])
            with col2_1:
                st.subheader('MobileNet accuracy :')
                st.dataframe(im_acc)
            with col2_2:
                st.subheader('Modèle textuel accuracy :')
                st.dataframe(txt_acc)

    with st.expander("RESULTATS"):
        nb_class_bullet_point = st.radio(
                "PLAN :",
                [
                    "**RESULTAT SUR 27 CLASSES**",
                    "**RESULTAT SUR 9 CLASSES**",
                ])
        if nb_class_bullet_point == "**RESULTAT SUR 27 CLASSES**":
            st.subheader("Classification report: ")
            st.image(f'/app/gallery/fusion_classification_report.png')
            st.subheader("Accuracy par classe :")
            st.image(f'/app/gallery/accuracy_fusion.png')
        elif nb_class_bullet_point == "**RESULTAT SUR 9 CLASSES**":
            st.subheader("Classification report: ")
            st.image(f'/app/gallery/fusion_classification_report_9_class.png')
            st.subheader("Accuracy par classe :")
            st.image(f'/app/gallery/accuracy_fusion_9_classes.png')

    with st.expander("DEPLOIEMENT DE LA SOLUTION"):
        col1, col2 = st.columns([1,2])
        with col1:
            st.error("Conflit sur nos environnements de travail respectif.")
            st.error("Incompatibilité entre la version de Tensorflow utilisée dans colab (`2.17`) et celle instalable en local (<`2.13`).")
            st.success("**Création d'une image Docker** à partir de celle de Tensorflow disponible dans le DockerHub.")
            st.success("**Bonus :** Possibilité de facilement déployer facilement le streamlit de démonstration sur le web.")
        with col2:
            st.image('/app/gallery/docker_logo.jpg')