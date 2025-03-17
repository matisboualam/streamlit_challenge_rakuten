import streamlit as st
import subprocess
import nltk
import logging

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def setup_nltk():
    logging.info("Setting up NLTK resources...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    logging.info("NLTK setup complete.")

@st.cache_resource
def setup_dvc():
    # Retrieve access key ID from Streamlit secrets
    logging.info("Retrieving access key ID...")
    access_key_id = st.secrets["dvc"]["access_id"]

    # List DVC remotes
    print("Listing DVC remotes...")
    subprocess.run(['dvc', 'remote', 'list'])

    # Modify DVC remote settings locally
    logging.info("Modifying DVC remote settings...")
    subprocess.run([
        'dvc', 'remote', 'modify', 'origin', '--local', 
        'access_key_id', f'{access_key_id}'
    ])

    subprocess.run([
        'dvc', 'remote', 'modify', 'origin', '--local', 
        'secret_access_key', f'{access_key_id}'
    ])

    # Pull latest data from DVC
    logging.info("Pulling latest data from DVC...")
    subprocess.run(['dvc', 'pull'])

@st.cache_data
def setup_environment():
    logging.info("Setting up environment...")

    # Call the cached setup functions
    setup_dvc()
    setup_nltk()

    logging.info("Environment setup complete.")

# Execute the function
setup_environment()
