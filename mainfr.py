import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# Charger les caractéristiques et les noms de fichiers
liste_de_caracteristiques = np.array(pickle.load(open('featurevector.pkl', 'rb')))
noms_de_fichiers = pickle.load(open('filenames.pkl', 'rb'))

# Initialiser le modèle ResNet50
modele = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
modele.trainable = False

# Créer un modèle séquentiel avec GlobalMaxPooling2D
modele = tensorflow.keras.Sequential([
    modele,
    GlobalMaxPooling2D()
])

# Contenu
st.title('Système de recommandation de mode homme & femme')

def sauvegarder_fichier_televerse(televersement_fichier):
    try:
        with open(os.path.join('uploads', televersement_fichier.name), 'wb') as f:
            f.write(televersement_fichier.getbuffer())
        return 1
    except:
        return 0

def extraire_caracteristique(chemin_image, modele):
    # Extraction des caractéristiques de l'image
    img = cv2.imread(chemin_image)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    resultat = modele.predict(pre_img).flatten()
    normalise = resultat / norm(resultat)
    return normalise

def recommander(caracteristiques, liste_de_caracteristiques):
    # Trouver les images les plus proches en fonction des caractéristiques extraites
    voisins = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    voisins.fit(liste_de_caracteristiques)
    distances, indices = voisins.kneighbors([caracteristiques])
    return indices

def file_exists(file_path):
    return os.path.isfile(file_path)

# Étape du téléversement de fichier
fichier_televerse = st.file_uploader(label='Téléversez votre fichier ici', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

if fichier_televerse is not None:
    if sauvegarder_fichier_televerse(fichier_televerse):
        # Affichage de l'image téléversée
        image_affichee = Image.open(fichier_televerse)
        image_redimensionnee = image_affichee.resize((200, 200))
        st.image(image_redimensionnee)

        # Extraction des caractéristiques de l'image téléversée
        caracteristiques = extraire_caracteristique(os.path.join("uploads", fichier_televerse.name), modele)

        # Recommandation des images similaires
        indices = recommander(caracteristiques, liste_de_caracteristiques)
        num_images = 10  # Number of images to display
        num_cols = 5  # Number of columns

        columns = st.columns(num_cols)  # Create columns

        for i in range(num_images):
            image_index = indices[0][i]
            image_path = noms_de_fichiers[image_index]

            # Check if the image file exists
            if file_exists(image_path):
                col_idx = i % num_cols
                with columns[col_idx]:  # Use context manager to place images in columns
                    st.image(image_path)
    else:
        st.error("Une erreur est survenue lors du téléversement du fichier")