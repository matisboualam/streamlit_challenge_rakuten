import streamlit as st

def conclusion():
    st.title("Conclusion et perspectives")
    with st.expander("BILAN"):
        st.subheader("Benchmark Performance : ")
        col1, col2 = st.columns([1,1])
        with col1:
            st.info("**Rakuten Benchmark model :** ")
            st.markdown(
            """
            | Image | Texte | Fusion |
            |-----------|-----------|-----------|
            | 0.5534  | 0.8113  | None  |
            """
            )
        with col2:
            st.info("**Nos modélisations :**")
            st.markdown(
            """
            | Image | Texte | Fusion |
            |-----------|-----------|-----------|
            | 0.5631  | 0.5665  | 0.6754  |
            """
            )
        st.subheader("Validation Des Objectifs : ")
        col1, col2 = st.columns([1,10])
        with col2:
            st.success("- **Modélisation d'une solution de classification** en s'appuyant sur **les titres et description** des articles issus du catalogue Rakuten.")
            st.success("- **Modélisation d'une solution de classification** en s'appuyant sur les **images** des articles issus du catalogue Rakuten.")
            st.success("- **Modélisation d'une méthode multimodale** permettant la classification des articles issus du catalogue Rakuten.")
            st.success("- **Intégration** de la solution développée sous la forme d'une **application Streamlit.**")

    with st.expander("AMELIORATIONS"):
        st.warning("**Modèlisation multimodale**")
        col1, col2 = st.columns([1,10])
        with col2:
            st.info('- concatenation des vecteurs d\'input (image/texte).')
            st.info('- entrainement de couches supplémentaires en sortie des features extraites.')
            
        st.warning("**Rendre l'interaction utilisateur de l'application Streamlit plus personnelle**")
        col1, col2 = st.columns([1,10])
        with col2:
            st.info('- intégration d\'un encadré permettant la rédaction d\'une description fictive.')
            st.info('- importation d\'une image depuis un lien url.')
            st.info('- classification sur l\'article fictif créé.')
            
        
# st.set_page_config(page_title="Rakuten Multimodal Classifier",
#                    page_icon="gallery/rakuten.svg",
#                    layout="wide")
# st.sidebar.radio("Go to", "foobar")
# conclusion()