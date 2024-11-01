import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
from collections import defaultdict, Counter
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate
from tensorflow.keras.models import Model


# Chemin vers les images
path = os.path.join(os.getcwd(), "art-challenge", "images_lq")

# Collecter les images et leurs informations
data = []
for image_file in os.listdir(path):
    artist = '_'.join(image_file.split('_')[:-1])
    img_path = os.path.join(path, image_file)
    with Image.open(img_path) as img:
        width, height = img.size
    data.append({
        'image_path': img_path,
        'artist': artist,
        'width': width,
        'height': height
    })

# Créer un DataFrame
df_images = pd.DataFrame(data)

artist_nationalities_path = 'art-challenge/artists.csv'
artist_nationalities_df = pd.read_csv(artist_nationalities_path)

df_images_nationalites = df_images

# Nettoyer et standardiser les noms d'artistes pour correspondre à la colonne "name" dans le CSV
df_images_nationalites['artist_cleaned'] = df_images_nationalites['artist'].str.replace('_', ' ')

# Fusionner les données en utilisant les noms des artistes pour ajouter la première nationalité
# en séparant au cas où plusieurs nationalités sont listées
artist_nationalities_df['nationality'] = artist_nationalities_df['nationality'].str.split(',').str[0]
df_images_nationalites = df_images_nationalites.merge(artist_nationalities_df[['name', 'nationality']], 
                            how='left', 
                            left_on='artist_cleaned', 
                            right_on='name')

# Supprimer les colonnes temporaires
df_images_nationalites.drop(columns=['artist_cleaned', 'name'], inplace=True)




def filter_artists(df, min_images=75, max_images=200):
    """
    Filtre le DataFrame en conservant uniquement les artistes ayant au moins 'min_images' images
    et réduit aléatoirement le nombre d'images à 'max_images' si un artiste en a plus.
    
    Args:
    - df: DataFrame contenant au moins une colonne 'artist' avec les noms des artistes.
    - min_images: Nombre minimum d'images qu'un artiste doit avoir pour être inclus.
    - max_images: Nombre maximum d'images autorisées par artiste. Si dépassé, des images sont retirées.

    Returns:
    - filtered_df: Nouveau DataFrame filtré.
    """
    # Compter le nombre d'images par artiste
    compte_artistes = Counter(df['artist'])

    # Filtrer les artistes ayant au moins 'min_images' images
    artists_to_keep = [artist for artist, count in compte_artistes.items() if count >= min_images]
    filtered_df = df[df['artist'].isin(artists_to_keep)].copy()

    # Réduire aléatoirement le nombre d'images des artistes ayant plus de 'max_images' images
    filtered_df = filtered_df.groupby('artist').apply(lambda x: x.sample(n=min(len(x), max_images), random_state=42)).reset_index(drop=True)

    return filtered_df

df_images_nationalites_filtre = filter_artists(df_images_nationalites)

df = df_images_nationalites_filtre
batch_size = 20
test_size = 0.3
img_size = (512,512)

label_encoder = LabelEncoder()
df['artist_encoded'] = label_encoder.fit_transform(df['artist'])

nationality_encoder = LabelEncoder()
df['nationality_encoded'] = nationality_encoder.fit_transform(df['nationality'])

# Diviser les données en ensembles d'entraînement et de test
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['artist_encoded'])

train_datagen = ImageDataGenerator(
    rotation_range = 40,
    rescale = 1./255,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode='nearest'
)

# Générateur pour l'ensemble de test (sans augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

from tensorflow.keras.preprocessing.image import load_img, img_to_array

def custom_data_generator(dataframe, datagen, batch_size, img_size, nationality_col, artist_col, augment=False):
    output_signature = (
        {
            'image_input': tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),  # Mettre en trois canaux
            'nationality_input': tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int64)
    )
    
    def generator():
        while True:
            batch = dataframe.sample(n=batch_size)
            images = []
            nationalities = []
            artists = []

            for _, row in batch.iterrows():
                # Charger l'image en mode RGB pour assurer 3 canaux
                img = load_img(row['image_path'], target_size=img_size, color_mode='rgb')
                img = img_to_array(img)
                
                if augment:
                    img = datagen.random_transform(img)
                
                img = img / 255.0  # Normalisation
                images.append(img)
                
                # Nationalité et artiste
                nationalities.append(row[nationality_col])
                artists.append(row[artist_col])

            yield {'image_input': np.array(images), 'nationality_input': np.array(nationalities).reshape(-1, 1)}, np.array(artists)

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)



# Générateur d'entraînement avec augmentation
train_gen = custom_data_generator(
    dataframe=train_df,
    datagen=train_datagen,
    batch_size=batch_size,
    img_size=img_size,
    nationality_col='nationality_encoded',
    artist_col='artist_encoded',
    augment=True
)

# Générateur de test sans augmentation
test_gen = custom_data_generator(
    dataframe=test_df,
    datagen=test_datagen,
    batch_size=batch_size,
    img_size=img_size,
    nationality_col='nationality_encoded',
    artist_col='artist_encoded',
    augment=False
)



# Paramètres du modèle
input_size = (512, 512, 3)
num_artists = df['artist_encoded'].nunique()



# Entrée Image
image_input = Input(shape=input_size, name='image_input')
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# Entrée Nationalité
nationality_input = Input(shape=(1,), name='nationality_input')
y = Dense(32, activation='relu')(nationality_input)

# Fusion des Caractéristiques
combined = concatenate([x, y])

# Couches Denses pour la Prédiction
z = Dense(64, activation='relu')(combined)
z = Dense(128, activation='relu')(z)
output = Dense(num_artists, activation='softmax', name='artist_output')(z)

# Modèle Final
model = Model(inputs=[image_input, nationality_input], outputs=output)

# Compilation du Modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Sauvegarde du modèle et des courbes de progression
model_save_path = os.path.join(os.getcwd(), 'best_model.h5')
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entraînement du modèle
epochs = 100

steps_per_epoch = len(train_df) // batch_size
validation_steps = len(test_df) // batch_size

history = model.fit(
    train_gen,
    validation_data=test_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[checkpoint]
)

# Tracé des courbes de progression
def plot_training_history(history):
    # Courbe de perte
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Courbe de précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Sauvegarde des courbes
    plt.savefig(os.path.join(os.getcwd(), 'training_progress.png'))
    plt.show()

# Tracer et sauvegarder les courbes
plot_training_history(history)

