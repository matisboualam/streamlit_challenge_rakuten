import streamlit as st
import subprocess
import nltk

def setup_environment():
    # Retrieve access key ID from Streamlit secrets
    print("Retrieving access key ID...")
    access_key_id = st.secrets["dvc"]["access_id"]
    print(f"Access Key ID: {access_key_id}")

    # List DVC remotes
    print("Listing DVC remotes...")
    subprocess.run(['dvc', 'remote', 'list'])

    # Modify DVC remote settings locally
    print("Modifying DVC remote settings...")
    subprocess.run([
        'dvc', 'remote', 'modify', 'origin', '--local', 
        'access_key_id', f'{access_key_id}'
    ])

    subprocess.run([
        'dvc', 'remote', 'modify', 'origin', '--local', 
        'secret_access_key', f'{access_key_id}'
    ])

    # Pull latest data from DVC
    print("Pulling latest data from DVC...")
    subprocess.run(['dvc', 'pull'])

    def setup_nltk():
        print("Setting up NLTK resources...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        print("NLTK setup complete.")

    setup_nltk()
    print("Environment setup complete.")

# Execute the function
setup_environment()
