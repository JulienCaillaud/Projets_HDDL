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

df_images = filter_artists(df_images)


# Fonction pour préparer le dataset
def prepare_dataset(df, img_size=(512, 512), test_size=0.3, batch_size=20):
    label_encoder = LabelEncoder()
    df['artist_encoded'] = label_encoder.fit_transform(df['artist'])
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['artist_encoded'])
    
    train_datagen = ImageDataGenerator(
        rotation_range=40, rescale=1./255, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='image_path', y_col='artist_encoded',
        target_size=img_size, batch_size=batch_size, class_mode='raw', shuffle=True
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col='image_path', y_col='artist_encoded',
        target_size=img_size, batch_size=batch_size, class_mode='raw', shuffle=False
    )
    
    return train_gen, test_gen, label_encoder

# Préparer le dataset avec augmentation
train_gen, test_gen, label_encoder = prepare_dataset(df_images)

# Construction du modèle CNN (exemple basique, ajustez selon vos besoins)
from tensorflow.keras import layers, models

num_classes = len(label_encoder.classes_)
input_shape = (512, 512, 3)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Sauvegarde du modèle et des courbes de progression
model_save_path = os.path.join(os.getcwd(), 'best_model.h5')
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entraînement du modèle
epochs = 100

batch_size = 20

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint],
	batch_size=batch_size
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

