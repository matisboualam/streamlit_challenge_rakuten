import nltk
import streamlit as st

@st.cache_data
def setup_nltk():
    nltk.data.path.append('/usr/share/nltk_data')
    nltk.download('stopwords', download_dir='/usr/share/nltk_data')
    nltk.download('wordnet', download_dir='/usr/share/nltk_data')
    nltk.download('punkt', download_dir='/usr/share/nltk_data')
setup_nltk()
