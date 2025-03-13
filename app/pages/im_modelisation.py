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


def im_model():
    img_dir = '/app/gallery'
    st.title("Modélisation - données images")
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
        with st.expander("**Formatage des CSV pour une manipulation optimale du dataset**"):
            # Create two columns for the layout
            col1, col2 = st.columns([1, 2])

            # Left column for textual description
            with col1:
                st.markdown("**Modifications apportées aux CSV :**")
                st.info('- Conversion de la colonne **`prdtypecode`** par leur nom de label associé.\n- Concaténation des colonnes **`product_id`** et **`image_id`** pour former la colonne **`image_path`**.\n- Fusion des dataframes **`X`** et **`Y`** pour conserver les modifications sur l\'ensemble des données.')

            # Right column for displaying dataframes
            with col2:
                # Split the right column into two subcolumns
                col2_1, col2_2, col2_3 = st.columns([2, 1, 3])

                with col2_1:
                    st.markdown("### `X` :")
                    st.dataframe(x_df.head())

                with col2_2:
                    st.markdown("### `Y` :")
                    st.dataframe(y_df.head())
                with col2_3:
                    st.markdown('### `dataset fusionné` :')
                    st.dataframe(df.head())


        with st.expander("**Inspection des images et analyse de la distribution des classes.**"):
            col1, col2 = st.columns([1,2])
            with col1:
                class_distrib = st.radio(
                    "",
                    [
                        "**INSPECTION DES IMAGES**",
                        "**DISTRIBUTION AVANT RÉÉQUILIBRAGE**",
                        "**DATASET RÉÉQUILIBRÉ**",
                    ])
                if class_distrib == "**INSPECTION DES IMAGES**":
                    st.markdown("**Toutes les images sont :**")
                    st.info('- sont associées à un article du dataset *(et inversement)*\n- au format **.jpg**.\n- de dimension **[3,500,500]**.\n- de distribution équivalente sur les canaux RGB.')                    
                elif class_distrib == "**DISTRIBUTION AVANT RÉÉQUILIBRAGE**":
                    st.markdown("**On observe une forte disparité de représentation des classes au sein du dataset global.**")
                    st.info("- Classe dominante : ***pool accessories (10209 éléments, <12%)***\n- Classe sous-représentée : ***figurines to paint and assemble (764 éléments, >1%)***\n- Technique de rééquilibrage du dataset : **réduire de manière aléatoire le nombre d'article par classe à seuil défini selon `nb_articles_souhaités`/`nb_classe`**")
                elif class_distrib == "**DATASET RÉÉQUILIBRÉ**":
                    st.markdown(
                            "| Taille du dataset réduit | **25000** images |\n"
                            "|-----------|-----------|\n"
                            "| Nombre de classe représentées | **27** |\n"
                            "| Nombre **max** d'articles par classe | **925** |\n"
                            "| Nombre **min** d'articles par classe | **764** |\n"
                        )
            with col2:
                if class_distrib == "**INSPECTION DES IMAGES**":
                    st.image(f'{img_dir}/rgb_distribution_class.png')
                elif class_distrib == "**DISTRIBUTION AVANT RÉÉQUILIBRAGE**":
                    st.image(f'{img_dir}/class_distribution_before_thresh.png', use_container_width=True)
                elif class_distrib == "**DATASET RÉÉQUILIBRÉ**":
                    st.image(f'{img_dir}/class_distribution_after_thresh.png')
                
        with st.expander("**Echantillons d'images par classe produit.**"):
            for productid, group in dataset.groupby('prdtypecode'):
                st.write(f"**Product ID: {productid}**")
                with st.container():
                    scrollable_images = st.columns(4)  # One column per image
                    for i in range(4):
                        image_path = f'/app/data/images/{group.image_path.iloc[i]}'
                        with scrollable_images[i]:
                            st.image(image_path, caption=f"Image {i+1}")
                    st.divider()

    if bullet_point == "**PREPROCESS**":
        with st.expander("**Techniques classiques de data augmentation.**"):
            st.markdown("Image transformation ❌ :\n- **Dataset suffisament volumineux et varié dans les représentations de ses classes.**\n- **Coût de temps de calcul très élévé sur 25000 images.**")
            st.image(f'{img_dir}/data_augment_illustration.png')
        with st.expander("**Manipulation des données pour correspondre au format d'input attendu.**"):
            st.markdown("Utilisation de la classe objet `ImageDataGenerator` issue de la librairie tensorflow keras :\n- **Modification des dimensions de l'image pour correspondre au format d'input requis. [3, 224, 224]**\n- **Normalisation des valeurs de pixels.**\n- **Manipulation du dataset en mémoire.**")
            st.image(f'{img_dir}/preprocess_scheme.png')

    if bullet_point == "**SOLUTION**":
        with st.expander("**Fonctionnement d'un CNN**"):
            st.image(f'{img_dir}/fonctionnement_cnn.jpg')
            st.markdown('- **Meilleure solution selon l\'état de l\'art pour interpréter la complexité du dataset.**')
        with st.expander("**Choix des CNN à entrainer**"):
            col1, col2 = st.columns([1,2])
            with col1:
                modele_bullet_point = st.radio(
                    "MODELS :",
                    [
                        "**RESNET**",
                        "**MOBILENET**",
                        "**VGGNET**"
                    ]
                )
                if modele_bullet_point == "**RESNET**":
                    st.markdown("""
                    - **Architecture :** Introduit des connexions résiduelles pour faciliter l'apprentissage des réseaux profonds.
                    - **Profondeur :** Très profond (ResNet-50, ResNet-101, ResNet-152).
                    - **Avantage :** Excellente performance sur des tâches complexes, évite le problème de vanishing gradients.
                    - **Limitation :** Plus lourd que MobileNet pour des appareils à faibles ressources.
                    """)
                if modele_bullet_point == "**MOBILENET**":
                    st.markdown("""
                    - **Architecture :** Conçu pour les appareils mobiles avec des convolutions séparables en profondeur.
                    - **Profondeur :** Plus léger que VGG ou ResNet.
                    - **Avantage :** Optimisé pour réduire la latence et la consommation énergétique.
                    - **Limitation :** Moins performant sur des tâches complexes.
                    """)
                if modele_bullet_point == "**VGGNET**":
                    st.markdown("""
                    - **Architecture :** Simple empilement de couches convolutives avec petits filtres \(3 \times 3\).
                    - **Profondeur :** 16 ou 19 couches (VGG16, VGG19).
                    - **Avantage :** Facile à comprendre et à implémenter.
                    - **Limitation :** Très lourd en termes de mémoire et de calculs.
                    """)
            with col2:
                if modele_bullet_point == "**RESNET**":
                    st.image(f'{img_dir}/resnet_arch.png')
                if modele_bullet_point == "**VGGNET**":
                    st.image(f'{img_dir}/vgg_arch.png')
                if modele_bullet_point == "**MOBILENET**":
                    st.image(f'{img_dir}/mobilenet_arch.jpg')

        with st.expander("**Conditions d'entrainement**"):
           st.markdown("""
                - **SPLIT :** 
                    - Séparation du jeu de données en jeu d'entrainement et de test (80/20).
                - **ENTRAINEMENT :** 
                    - Adaptation du model de base par ajout en cascade de couches denses pour aboutir à une prédiction sur nos 27 classes.
                    - Entrainement sur **20 epochs** avec les couches de base figées afin que les modèles apprennent à partir des features extraites issues du préentrainement sur le dataset `imagenet`.
                    - Fine-Tuning sur **10 epochs** par dégel progressif des couches de convolution.
                - **CALLBACKS :** 
                    - Définitions de callbacks permettant :
                        - l'adaptation du learning rate au cours de l'entrainement.
                        - l'arrêt de l'entrainement lorsque les modèles n'apprennent plus au fil des epochs.
                        - la sauvegarde des modèles lorsque que l'on note une amélioration sur les metriques de test.
                """)

    if bullet_point == "**RESULTAT**":
        with st.expander("**Suivi de l'entrainement**"):
            plt.figure(figsize=(12, 4))

            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'], label='Train Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.axvline(x=15, color='red', linestyle='--')  # Add vertical line
            plt.title('Model Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            # Ajouter un texte pour préciser "finetuning"
            plt.text(11, max(history['accuracy'])-0.018, 'Transfer\nLearning', color='blue', fontsize=10)
            plt.text(16, max(history['accuracy']), 'Finetuning', color='red', fontsize=10)

            plt.legend()

            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.axvline(x=15, color='red', linestyle='--')  # Add vertical line
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plt.text(11, max(history['val_loss']), 'Transfer\nLearning', color='blue', fontsize=10)
            plt.text(16, max(history['val_loss']), 'Finetuning', color='red', fontsize=10)

            plt.legend()

            plt.tight_layout()
            st.pyplot(plt)

        with st.expander("**Rapport de test**"):
            st.info('**CONFUSION MATRIX :**')
            st.image(f'{img_dir}/confusion_matrix_MobileNetV2.png')
            st.info('**ACCURACY SCORE :**')
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.info(f'avg accuracy : 0.5631')
                st.dataframe(accuracy[:7])
            with col2:
                st.dataframe(accuracy[7:17])
            with col3:
                st.dataframe(accuracy[17:])
                        

# st.set_page_config(page_title="Rakuten Multimodal Classifier",
#                    page_icon="gallery/rakuten.svg",
#                    layout="wide")
# st.sidebar.radio("Go to", "foobar")
# im_model()



