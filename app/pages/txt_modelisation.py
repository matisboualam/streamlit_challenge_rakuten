import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def load_dataset():
    x_df = pd.read_csv('/app/data/X_train_update.csv', index_col=0)
    y_df = pd.read_csv('/app/data/Y_train_CVw08PX.csv', index_col=0)
    dataset = pd.read_csv('/app/data/dataset.csv')
    df = dataset.drop(columns=['text'])
    history = pd.read_csv('/app/data/MobileNetV2_20_epochs_bal_25000.csv')
    history_ft = pd.read_csv('/app/data/MobileNetV2_20_epochs_bal_25000_FT_10_epochs.csv')
    hist = pd.concat([history, history_ft], ignore_index=True)
    accuracy = pd.read_csv('/app/data/accuracy/im_accuracy_MobileNet.csv')
    return x_df, y_df, df, dataset, hist, accuracy


def txt_model():
    img_dir = '/app/gallery'
    st.title("Modélisation - données textuelles")
    st.divider()
    x_df, y_df, df, dataset, history, accuracy = load_dataset()
    bullet_point = st.radio(
                "PLAN :",
                [
                    "**VISUALISATION**",
                    "**PREPROCESS**",
                    "**SOLUTION**",
                    "**RESULTAT**",
                ])
    if bullet_point == "**VISUALISATION**":
        with st.expander("**Inspection des csv**"):
            # Create two columns for the layout
            col1, col2 = st.columns([1, 2])

            # Left column for textual description
            with col1:
                st.markdown("**Inspection des colonnes `designation` et `description` :**")
                st.info('- On observe des **valeurs manquantes** pour la colonne `description`.')
                st.success('- Solution : jointure sur les colonnes `designation` et `description`.')

            # Right column for displaying dataframes
            with col2:
                st.markdown("### `X` :")
                st.dataframe(x_df.head())

        with st.expander("**Distribution des classes a sein du jeu de données.**"):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.info('Métrique de coefficient de variation **[CV = σ/μ]** avec σ est l\'écart-type des données et μ est la moyenne des données:\n - **jeu déséquilibré :** CV = 0.69\n- **jeu équilibré :** CV = 0.29')
                st.warning('**Avant Rééquilibrage :** Réduction de 75 % pour le jeu de donnée entrainé avec scikit-learn\n Note : le jeu de données pour Deep Learning n’est pas réduit')
                st.warning('**Après Rééquilibrage :** Reduction augmentation classes sur et sous représentées à une moyenne des autres classes')
            with col2:
                st.subheader('Avant Rééquilibrage')
                st.image(f'{img_dir}/txt_class_distrib_avant.png')
            with col3:
                st.subheader('Après Rééquilibrage')
                st.image(f'{img_dir}/txt_class_distrib_apres.png')

    if bullet_point == "**PREPROCESS**":
        col1, col2 = st.columns([1,1])
        with col1:
            st.info('**Techniques de data cleaning :**\n- Utilisation d’expressions régulières\n    - **r"\b[a-zA-Z]’"** : Identifie et supprime une lettre isolée suivie d\'une apostrophe\n   - **r’\d+’** : Repère et élimine toutes les occurrences de chiffres\n   - **r\'\W+’** : remplace une ou plusieurs occurrences consécutives de caractères non alphabétiques par un espace')
            st.info('**Techniques de linguistic normalisation :**\n- Lemmatization : réduction au lemme\n- suppression des stopwords :  élimine les mots non significatifs pour l\'analyse')
            st.info('**Représentation vectorielle des mots avec Word2Vec :**\n- Utilisation d’un modèle Word2Vec préentraîné pour convertir chaque mot en vecteur numérique, encapsulant son information sémantique.')
            st.info('**Moyennage des vecteurs pour une représentation globale :**\n- Pour chaque élément textuel, les vecteurs de tous les mots sont moyennés, créant un vecteur unique représentant la sémantique complète de la description.')

        with col2:
            st.image(f'{img_dir}/rpz_text_vec.png')

    if bullet_point == "**SOLUTION**":
        col1, col2 = st.columns([1,1])
        with col2:
            modele_bullet_point = st.radio(
                    "MODELS :",
                    [
                        "**RANDOM FOREST**",
                        "**ONE VS ALL AVEC SVM**",
                        "**DEEPLEARNING (GRU + CONV)**"
                    ]
                )
            if modele_bullet_point == "**RANDOM FOREST**":
                st.subheader('Random forest architecture :')
                st.image(f'{img_dir}/random_forest.png')
            if modele_bullet_point == "**ONE VS ALL AVEC SVM**":
                st.subheader('One vs all architecture :')
                st.image(f'{img_dir}/one_vs_all.png')
            if modele_bullet_point == "**DEEPLEARNING (GRU + CONV)**":
                st.subheader('Gru layer :')
                st.image(f'{img_dir}/gru_layer.png')

        with col1:
            with st.expander("**Description**"):
                if modele_bullet_point == "**RANDOM FOREST**":
                    st.markdown("""
                    - **Architecture :** Ensemble d'arbres de décision indépendants (bagging). Classification par majorité.
                    - **Avantage :** 
                        - Robuste face au surapprentissage.
                        - Bonne gestion des données manquantes et bruitées.
                    - **Limitation :** 
                        - Modèle lourd, coûteux en temps et en mémoire.
                        - Moins efficace pour les relations complexes (ex : texte long).
                    """)
                if modele_bullet_point == "**ONE VS ALL AVEC SVM**":
                    st.markdown("""
                    - **Architecture :** Classificateurs binaires pour chaque classe, choisit la classe avec la sortie la plus élevée.
                    - **Avantage :** 
                        - Efficace pour les petites données.
                        - Haute précision pour les marges séparables.
                    - **Limitation :** 
                        - Long à entraîner avec de nombreuses classes.
                        - Difficile à appliquer sur des données non linéaires complexes.
                    """)
                if modele_bullet_point == "**DEEPLEARNING (GRU + CONV)**":
                    st.markdown("""
                    - **Architecture :** Couches GRU bidirectionnelles et convolution pour capturer les relations séquentielles et locales, attention multi-tête pour focalisation sur les parties clés.
                    - **Avantage :** 
                        - Excellente gestion des séquences complexes.
                        - Performances supérieures sur des données non structurées.
                    - **Limitation :** 
                        - Nécessite plus de données et de puissance de calcul.
                        - Risque de surapprentissage sans régularisation.
                    """)
            with st.expander("**Conditions d'entrainement**"):
                if modele_bullet_point == "**RANDOM FOREST**":
                    st.markdown("""
                    - **Noyau :** RBF (Radial Basis Function), C=0.1, random_state=42. Le noyau RBF permet de capturer des relations non linéaires entre les données C=0.1 : 
                        - Une régularisation plus faible permet au modèle de mieux généraliser, bien qu'il puisse réduire son ajustement aux données.
                    """)
                if modele_bullet_point == "**ONE VS ALL AVEC SVM**":
                    st.markdown("""
                    - **Paramètres :**
                        - max_depth=20, 
                        - max_features='sqrt', 
                        - n_estimators=50, 
                        - min_samples_leaf=1, 
                        - min_samples_split=2.
                    """)
                if modele_bullet_point == "**DEEPLEARNING (GRU + CONV)**":
                    st.markdown("""
                    - **Optimisation :**
                        - Adam avec learning_rate=0.001, loss='categorical_crossentropy', metrics=['accuracy'].
                    - **Callbacks :** 
                        - ReduceLROnPlateau, EarlyStopping, ModelCheckpoint.
                    """)

    if bullet_point == "**RESULTAT**":
        with st.expander("**ROC**"):
            st.image(f'{img_dir}/roc_txt.png')
            st.image(f'{img_dir}/roc_txt_micro.png')

        with st.expander("**Matrices de confusion**"):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image(f'{img_dir}/mat_conf_random.png')
            with col2:
                st.image(f'{img_dir}/mat_conf_svm.png')
            with col3:
                st.image(f'{img_dir}/mat_conf_deep.png')
        with st.expander("**Mean accuracy**"):
            st.image(f'{img_dir}/txt_compare_models.png')
                        

# st.set_page_config(page_title="Rakuten Multimodal Classifier",
#                    page_icon="gallery/rakuten.svg",
#                    layout="wide")
# st.sidebar.radio("Go to", "foobar")
# txt_model()



