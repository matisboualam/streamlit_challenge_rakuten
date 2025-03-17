import streamlit as st
import subprocess
import nltk

def setup_environment():
    # Retrieve access key ID from Streamlit secrets
    st.write("Retrieving access key ID...")
    access_key_id = st.secrets["dvc"]["access_id"]
    print(f"Access Key ID: {access_key_id}")

    # List DVC remotes
    st.write("Listing DVC remotes...")
    subprocess.run(['dvc', 'remote', 'list'])

    # Modify DVC remote settings locally
    st.write("Modifying DVC remote settings...")
    subprocess.run([
        'dvc', 'remote', 'modify', 'origin', '--local', 
        'access_key_id', f'{access_key_id}'
    ])

    subprocess.run([
        'dvc', 'remote', 'modify', 'origin', '--local', 
        'secret_access_key', f'{access_key_id}'
    ])

    # Pull latest data from DVC
    st.write("Pulling latest data from DVC...")
    subprocess.run(['dvc', 'pull'])

    # Setup NLTK resources
    @st.cache_data
    def setup_nltk():
        st.write("Setting up NLTK resources...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        st.write("NLTK setup complete.")

    setup_nltk()
    st.write("Environment setup complete.")

# Execute the function
setup_environment()
