import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
from collections import Counter
from collections import defaultdict

path = os.getcwd()
path = os.path.join(path, "art-challenge", "images_lq")

# Obtenir la liste de toutes les images dans le dossier
list_images = os.listdir(path)
data = []

# Boucle à travers chaque image
for image_file in list_images:
    # Séparer l'artiste du reste du nom de fichier (enlever l'extension)
    artiste = '_'.join(image_file.split('_')[:-1])  # Récupérer tout avant le dernier underscore "_"
    
    # Lire l'image
    img = plt.imread(os.path.join(path, image_file))
    
    # Ajouter l'image et le nom complet de l'artiste au dataset
    data.append([img, artiste])

def convertir_images_en_rgb(data):
    """
    Convertit toutes les images du dataset au format RGB si elles ne le sont pas déjà.

    Args:
    - data: Liste contenant des paires [image, artiste].

    Returns:
    - data_rgb: Liste contenant des paires [image en RGB, artiste].
    """
    data_rgb = []
    
    for img, artiste in data:
        # Si l'image est en niveaux de gris (2 dimensions), la convertir en RGB
        if len(img.shape) == 2:
            # Dupliquer les valeurs pour obtenir 3 canaux RGB
            img_rgb = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 4:
            # Si l'image a un canal alpha (RGBA), enlever le canal alpha
            img_rgb = img[:, :, :3]
        else:
            # Si l'image est déjà en RGB, on la garde telle quelle
            img_rgb = img
        
        # Ajouter l'image convertie (ou non) au dataset
        data_rgb.append([img_rgb, artiste])
    
    return data_rgb

# Exemple d'utilisation
data = convertir_images_en_rgb(data)

def enregistrer_data(data, nom_fichier="data.npy"):
    """
    Enregistre le dataset dans un fichier .npy à un chemin spécifié.

    Args:
    - data: Liste contenant des paires [image, artiste].
    - nom_fichier: Nom du fichier de sortie (par défaut 'data.npy').
    """
    # Spécifier le chemin d'enregistrement
    path = "/home/regnie/Bureau/5A/Projets/Mini-projet CNN"
    
    # S'assurer que le répertoire existe
    if not os.path.exists(path):
        os.makedirs(path)  # Créer le répertoire s'il n'existe pas
    
    # Convertir la liste en un tableau NumPy pour l'enregistrement
    data_array = np.array(data, dtype=object)
    
    # Enregistrer le dataset dans le fichier
    np.save(os.path.join(path, nom_fichier), data_array)
    print(f"Dataset enregistré sous {nom_fichier} dans le répertoire {path}")

# Exemple d'utilisation
enregistrer_data(data)


