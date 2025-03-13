import streamlit as st

def home():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("")
        st.image("gallery/rakuten.svg", use_container_width='auto')
        st.title("")
        st.divider()
        st.markdown('##### Eleve 1 : Emmanuel Dumery\n##### Eleve 2 : Matis Boualam\n##### Mentorat : Gaspard')
