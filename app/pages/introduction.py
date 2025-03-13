import streamlit as st

def introduction():
    st.title("Introduction")
    st.header("Mise en contexte")
    with st.expander("PROJET"):
        col1, col2 = st.columns([1,2])
        with col1:
            st.info("Ce projet vise à développer un modèle de classification multimodale (texte et image) à grande échelle pour la plateforme e-commerce de Rakuten France.")
            st.info("L’objectif est de prédire avec précision les codes de type de produit en se basant sur les titres, les images et les descriptions des produits.")
        with col2:
            st.image('/app/gallery/scheme_multimodal_classification.png')
    with st.expander("MOTIVATIONS"):
        col1, col2 = st.columns([1,1])
        with col1:
            st.info("Projet en lien avec nos activités en entreprise :")
            st.markdown("""
                        - **Computer vision (Calibration de camera, manipulation d'images ...)** 
                        - **Entrainement de modèle de détection d'objet (type YOLO)**
                        """)
            st.info("Projet interessant sur le plan multimodal :")
            st.markdown("""
                        - **Se confronter à différents formats de données** 
                        - **Manipulation d'un jeu de données de taille conséquente.**
                        - **Utilisation de modèles aux architectures complexes**
                        """)
        with col2:
            st.image('/app/gallery/yolo_example.png')

    with st.expander("AVANCEMENT DU PROJET"):
        st.success("ETAPE 1 : ")
        st.markdown("""
                    - **Inspection de la plateforme Rakuten** 
                    - **Identification de profil d'utilisateur type et des problèmes auxquels ils font face.**
                    - **Proposition d'amélioration à apporter et définition d'une solution produit**
                    """)
        st.info("ETAPE 2 : ")
        st.markdown("""
                    - **Redéfinition de la solution en fonction des données mises à disposition** 
                    - **Inspection du dataset sur lequel on s'appuiera pour mettre en place la solution produit.**
                    - **Modélisation de la solution.**
                    - **Intégration de la solution sous la forme d'une application Streamlit.**
                    """)
        st.warning("ETAPE 3 : ")
        st.markdown("""
                    - **Déploiement de la solution.**
                    - **...**
                    """)



# st.set_page_config(page_title="Rakuten Multimodal Classifier",
#                    page_icon="gallery/rakuten.svg",
#                    layout="wide")
# st.sidebar.radio("Go to", "foobar")
# introduction()