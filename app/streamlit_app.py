import streamlit as st

st.set_page_config(page_title="Rakuten Multimodal Classifier",
                   page_icon="gallery/rakuten.svg",
                   layout="wide")

from pages.home import home
from pages.introduction import introduction
from pages.rakuten_presentation import rakuten_prez
from pages.im_modelisation import im_model
from pages.txt_modelisation import txt_model
from pages.fusion_modelisation import fusion_model
from pages.demo import demo
from pages.conclusion import conclusion

st.title("Projet Rakuten\n## Étape 2 : Visualisation - Modélisation")
st.divider()

pages = {
    "Home": home,
    "Introduction": introduction,
    "Presentation Rakuten": rakuten_prez,
    "Modelisation Image": im_model,
    "Modelisation Texte": txt_model,
    "Modelisation Fusion": fusion_model,
    "Demonstration": demo,
    "Conclusion": conclusion,
}
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Render the selected page
page_function = pages[selection]
page_function()
