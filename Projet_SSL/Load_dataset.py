import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from sklearn.metrics import roc_curve, auc
import random
import numpy as np
from torch.utils.data import Subset
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_autoVI(imput_size = 128):
    autovi_train_dir = "AutoVI/engine_wiring/train"
    autovi_test_dir = "AutoVI/engine_wiring/test"

    autovi_train_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    autovi_test_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    autovi_train_dataset = datasets.ImageFolder(
        root=autovi_train_dir,
        transform=autovi_train_transform
    )

    autovi_test_dataset = datasets.ImageFolder(
        root=autovi_test_dir,
        transform=autovi_test_transform
    )
    
    autovi_train_loader = DataLoader(autovi_train_dataset, batch_size=16, shuffle=True)
    autovi_test_loader = DataLoader(autovi_test_dataset, batch_size=16, shuffle=False)

    good_class_idx = autovi_test_dataset.class_to_idx['good']

    good_indices = [i for i, (_, label) in enumerate(autovi_test_dataset) if label == good_class_idx]

    filtered_autovi_test_dataset = Subset(autovi_test_dataset, good_indices)

    filtered_autovi_test_loader = DataLoader(filtered_autovi_test_dataset, batch_size=16, shuffle=False)

        # Filtrer les anomalies (non-"good")
    anomaly_indices = [i for i, (_, label) in enumerate(autovi_test_dataset) if label != good_class_idx]
    anomaly_autovi_test_dataset = Subset(autovi_test_dataset, anomaly_indices)
    anomaly_autovi_test_loader = DataLoader(anomaly_autovi_test_dataset, batch_size=16, shuffle=False)

   
    return autovi_train_dataset, autovi_test_dataset, autovi_train_loader, autovi_test_loader, filtered_autovi_test_dataset, filtered_autovi_test_loader,anomaly_autovi_test_dataset, anomaly_autovi_test_loader


def load_bottle(imput_size = 128):
    bottle_train_dir = "MVTecAD/bottle/bottle/train"
    bottle_test_dir = "MVTecAD/bottle/bottle/test"

    bottle_train_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    bottle_test_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    bottle_train_dataset = datasets.ImageFolder(
        root=bottle_train_dir,
        transform=bottle_train_transform
    )

    bottle_test_dataset = datasets.ImageFolder(
        root=bottle_test_dir,
        transform=bottle_test_transform
    )
    
    bottle_train_loader = DataLoader(bottle_train_dataset, batch_size=16, shuffle=True)
    bottle_test_loader = DataLoader(bottle_test_dataset, batch_size=16, shuffle=False)

    good_class_idx = bottle_test_dataset.class_to_idx['good']

    good_indices = [i for i, (_, label) in enumerate(bottle_test_dataset) if label == good_class_idx]

    filtered_bottle_test_dataset = Subset(bottle_test_dataset, good_indices)

    filtered_bottle_test_loader = DataLoader(filtered_bottle_test_dataset, batch_size=16, shuffle=False)

            # Filtrer les anomalies (non-"good")
    anomaly_indices = [i for i, (_, label) in enumerate(bottle_test_dataset) if label != good_class_idx]
    anomaly_bottle_test_dataset = Subset(bottle_test_dataset, anomaly_indices)
    anomaly_bottle_test_loader = DataLoader(anomaly_bottle_test_dataset, batch_size=16, shuffle=False)
    
    return bottle_train_dataset, bottle_test_dataset, bottle_train_loader, bottle_test_loader, filtered_bottle_test_dataset, filtered_bottle_test_loader,anomaly_bottle_test_dataset,anomaly_bottle_test_loader

def load_capsule(imput_size = 128):
    capsule_train_dir = "MVTecAD/capsule/capsule/train"
    capsule_test_dir = "MVTecAD/capsule/capsule/test"

    capsule_train_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    capsule_test_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    capsule_train_dataset = datasets.ImageFolder(
        root=capsule_train_dir,
        transform=capsule_train_transform
    )

    capsule_test_dataset = datasets.ImageFolder(
        root=capsule_test_dir,
        transform=capsule_test_transform
    )
    
    capsule_train_loader = DataLoader(capsule_train_dataset, batch_size=16, shuffle=True)
    capsule_test_loader = DataLoader(capsule_test_dataset, batch_size=16, shuffle=False)

    good_class_idx = capsule_test_dataset.class_to_idx['good']

    good_indices = [i for i, (_, label) in enumerate(capsule_test_dataset) if label == good_class_idx]

    filtered_capsule_test_dataset = Subset(capsule_test_dataset, good_indices)

    filtered_capsule_test_loader = DataLoader(filtered_capsule_test_dataset, batch_size=16, shuffle=False)

                # Filtrer les anomalies (non-"good")
    anomaly_indices = [i for i, (_, label) in enumerate(capsule_test_dataset) if label != good_class_idx]
    anomaly_capsule_test_dataset = Subset(capsule_test_dataset, anomaly_indices)
    anomaly_capsule_test_loader = DataLoader(anomaly_capsule_test_dataset, batch_size=16, shuffle=False)
   
    return capsule_train_dataset, capsule_test_dataset, capsule_train_loader, capsule_test_loader, filtered_capsule_test_dataset, filtered_capsule_test_loader,anomaly_capsule_test_dataset,anomaly_capsule_test_loader


def load_hazelnut(imput_size = 128):
    hazelnut_train_dir = "MVTecAD/hazelnut/hazelnut/train"
    hazelnut_test_dir = "MVTecAD/hazelnut/hazelnut/test"

    hazelnut_train_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    hazelnut_test_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    hazelnut_train_dataset = datasets.ImageFolder(
        root=hazelnut_train_dir,
        transform=hazelnut_train_transform
    )

    hazelnut_test_dataset = datasets.ImageFolder(
        root=hazelnut_test_dir,
        transform=hazelnut_test_transform
    )
    
    hazelnut_train_loader = DataLoader(hazelnut_train_dataset, batch_size=16, shuffle=True)
    hazelnut_test_loader = DataLoader(hazelnut_test_dataset, batch_size=16, shuffle=False)

    good_class_idx = hazelnut_test_dataset.class_to_idx['good']

    good_indices = [i for i, (_, label) in enumerate(hazelnut_test_dataset) if label == good_class_idx]

    filtered_hazelnut_test_dataset = Subset(hazelnut_test_dataset, good_indices)

    filtered_hazelnut_test_loader = DataLoader(filtered_hazelnut_test_dataset, batch_size=16, shuffle=False)

    anomaly_indices = [i for i, (_, label) in enumerate(hazelnut_test_dataset) if label != good_class_idx]
    anomaly_hazelnut_test_dataset = Subset(hazelnut_test_dataset, anomaly_indices)
    anomaly_hazelnut_test_loader = DataLoader(anomaly_hazelnut_test_dataset, batch_size=16, shuffle=False)
   
    return hazelnut_train_dataset, hazelnut_test_dataset, hazelnut_train_loader, hazelnut_test_loader, filtered_hazelnut_test_dataset, filtered_hazelnut_test_loader,anomaly_hazelnut_test_dataset,anomaly_hazelnut_test_loader


def load_toothbrush(imput_size = 128):
    toothbrush_train_dir = "MVTecAD/toothbrush/toothbrush/train"
    toothbrush_test_dir = "MVTecAD/toothbrush/toothbrush/test"

    toothbrush_train_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    toothbrush_test_transform = transforms.Compose([
        transforms.Resize((imput_size, imput_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    toothbrush_train_dataset = datasets.ImageFolder(
        root=toothbrush_train_dir,
        transform=toothbrush_train_transform
    )

    toothbrush_test_dataset = datasets.ImageFolder(
        root=toothbrush_test_dir,
        transform=toothbrush_test_transform
    )
    
    toothbrush_train_loader = DataLoader(toothbrush_train_dataset, batch_size=16, shuffle=True)
    toothbrush_test_loader = DataLoader(toothbrush_test_dataset, batch_size=16, shuffle=False)

    good_class_idx = toothbrush_test_dataset.class_to_idx['good']

    good_indices = [i for i, (_, label) in enumerate(toothbrush_test_dataset) if label == good_class_idx]

    filtered_toothbrush_test_dataset = Subset(toothbrush_test_dataset, good_indices)

    filtered_toothbrush_test_loader = DataLoader(filtered_toothbrush_test_dataset, batch_size=16, shuffle=False)

    anomaly_indices = [i for i, (_, label) in enumerate(toothbrush_test_dataset) if label != good_class_idx]
    anomaly_toothbrush_test_dataset = Subset(toothbrush_test_dataset, anomaly_indices)
    anomaly_toothbrush_test_loader = DataLoader(anomaly_toothbrush_test_dataset, batch_size=16, shuffle=False)
   
    return toothbrush_train_dataset, toothbrush_test_dataset, toothbrush_train_loader, toothbrush_test_loader, filtered_toothbrush_test_dataset, filtered_toothbrush_test_loader,anomaly_toothbrush_test_dataset,anomaly_toothbrush_test_loader

"""
autovi_train_dataset, autovi_test_dataset, autovi_train_loader, autovi_test_loader, filtered_autovi_test_dataset, filtered_autovi_test_loader = load_autoVI()
print(f"Classes dans le train (AutoVI) : {autovi_train_dataset.classes}")
print(f"Classes dans le test (AutoVI) : {autovi_test_dataset.classes}")
print(f"Nombre d'images dans le train (AutoVI) : {len(autovi_train_dataset)}")
print(f"Nombre d'images dans le test (AutoVI) : {len(autovi_test_dataset)}")
print(f"Nombre d'images de la classe 'good' dans le test : {len(filtered_autovi_test_dataset)}")
print(" ")

bottle_train_dataset, bottle_test_dataset, bottle_train_loader, bottle_test_loader, filtered_bottle_test_dataset, filtered_bottle_test_loader = load_bottle()
print(f"Classes dans le train (Bottle) : {bottle_train_dataset.classes}")
print(f"Classes dans le test (Bottle) : {bottle_test_dataset.classes}")
print(f"Nombre d'images dans le train (Bottle) : {len(bottle_train_dataset)}")
print(f"Nombre d'images dans le test (Bottle) : {len(bottle_test_dataset)}")
print(f"Nombre d'images de la classe 'good' dans le test : {len(filtered_bottle_test_dataset)}")
print(" ")

capsule_train_dataset, capsule_test_dataset, capsule_train_loader, capsule_test_loader, filtered_capsule_test_dataset, filtered_capsule_test_loader = load_capsule()
print(f"Classes dans le train (Capsule) : {capsule_train_dataset.classes}")
print(f"Classes dans le test (Capsule) : {capsule_test_dataset.classes}")
print(f"Nombre d'images dans le train (Capsule) : {len(capsule_train_dataset)}")
print(f"Nombre d'images dans le test (Capsule) : {len(capsule_test_dataset)}")
print(f"Nombre d'images de la classe 'good' dans le test : {len(filtered_capsule_test_dataset)}")
print(" ")

hazelnut_train_dataset, hazelnut_test_dataset, hazelnut_train_loader, hazelnut_test_loader, filtered_hazelnut_test_dataset, filtered_hazelnut_test_loader = load_hazelnut()
print(f"Classes dans le train (Hazelnut) : {hazelnut_train_dataset.classes}")
print(f"Classes dans le test (Hazelnut) : {hazelnut_test_dataset.classes}")
print(f"Nombre d'images dans le train (Hazelnut) : {len(hazelnut_train_dataset)}")
print(f"Nombre d'images dans le test (Hazelnut) : {len(hazelnut_test_dataset)}")
print(f"Nombre d'images de la classe 'good' dans le test (Hazelnut) : {len(filtered_hazelnut_test_dataset)}")
print(" ")

toothbrush_train_dataset, toothbrush_test_dataset, toothbrush_train_loader, toothbrush_test_loader, filtered_toothbrush_test_dataset, filtered_toothbrush_test_loader = load_toothbrush()
print(f"Classes dans le train (Toothbrush) : {toothbrush_train_dataset.classes}")
print(f"Classes dans le test (Toothbrush) : {toothbrush_test_dataset.classes}")
print(f"Nombre d'images dans le train (Toothbrush) : {len(toothbrush_train_dataset)}")
print(f"Nombre d'images dans le test (Toothbrush) : {len(toothbrush_test_dataset)}")
print(f"Nombre d'images de la classe 'good' dans le test (Toothbrush) : {len(filtered_toothbrush_test_dataset)}")
print(" ")
"""