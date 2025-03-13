import streamlit as st

def rakuten_prez():
    st.title("Challenge Rakuten")
    with st.expander("PLATEFORME CHALLENGE DATA"):
        st.info('- 99k articles répertoriés dans un dossier images et 3 csv divisés en jeu d\'entrainement et de test.')
        st.info('- Données appartenant à Rakuten France dont l\'utilisation n\'est pas commerciale et mise à disposition pour la durée du challenge Rakuten')
        st.image('/app/gallery/challenge_data_web.png')
        
# st.set_page_config(page_title="Rakuten Multimodal Classifier",
#                    page_icon="gallery/rakuten.svg",
#                    layout="wide")
# st.sidebar.radio("Go to", "foobar")
# rakuten_prez()

