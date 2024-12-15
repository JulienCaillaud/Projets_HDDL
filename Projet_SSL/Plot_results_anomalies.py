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
import warnings
from torch.utils.data import Subset
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torch.nn.functional import mse_loss
import pickle

def plot_histograms_for_dataset(results, dataset_name):
    """
    Trace les histogrammes des scores négatifs et positifs pour chaque modèle d'un dataset donné.
    Les subplots sont alignés côte à côte.
    
    Args:
        results (dict): Le dictionnaire contenant les résultats pour chaque dataset et modèle.
        dataset_name (str): Le nom du dataset à visualiser.
    """
    if dataset_name not in results:
        raise ValueError(f"Dataset {dataset_name} n'existe pas dans les résultats.")

    model_results = results[dataset_name]
    num_models = len(model_results)

    # Créer des subplots avec 1 ligne et autant de colonnes que de modèles
    fig, axes = plt.subplots(1, num_models, figsize=(num_models * 5, 5), sharey=True)

    # Si un seul modèle, convertir `axes` en une liste pour une manipulation cohérente
    if num_models == 1:
        axes = [axes]

    for idx, (model_type, data) in enumerate(model_results.items()):
        ax = axes[idx]

        # Récupérer les scores négatifs et positifs
        scores_negatives = data["scores_negatives"]
        scores_positives = data["scores_positives"]

        # Tracer les histogrammes
        ax.hist(scores_negatives, bins=50, alpha=0.5, label='Good Samples', color='blue')
        ax.hist(scores_positives, bins=50, alpha=0.5, label='Anomalies', color='orange')
        ax.axvline(data["thresholds"], color='red', linestyle='--', label=f'Threshold = {data["thresholds"]:.4f}')

        # Ajouter les titres et légendes
        ax.set_title(f"{dataset_name} - {model_type}")
        ax.set_xlabel("MSE Score")
        ax.legend()

    # Ajouter un label commun pour l'axe Y
    fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical")

    # Ajuster les espacements entre les subplots
    plt.tight_layout()
    plt.show()


def plot_roc_curves_for_dataset(results, dataset_name):
    """
    Trace les courbes ROC pour chaque modèle associé à un dataset donné.
    Les trois premiers subplots montrent une courbe ROC par modèle.
    Le quatrième subplot combine toutes les courbes ROC sans afficher les AUC.
    
    Args:
        results (dict): Le dictionnaire contenant les résultats pour chaque dataset et modèle.
        dataset_name (str): Le nom du dataset à visualiser.
    """
    if dataset_name not in results:
        raise ValueError(f"Dataset {dataset_name} n'existe pas dans les résultats.")

    model_results = results[dataset_name]
    num_models = len(model_results)

    if num_models > 3:
        raise ValueError("Cette fonction est conçue pour un maximum de 3 modèles par dataset.")

    # Créer des subplots : 4 colonnes (3 modèles + 1 pour la combinaison)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    # Couleurs pour les modèles
    colors = ['blue', 'green', 'orange']

    # Tracer les courbes ROC pour chaque modèle
    for idx, (model_type, data) in enumerate(model_results.items()):
        ax = axes[idx]
        fpr = data["fpr"]
        tpr = data["tpr"]
        roc_auc = data["roc_auc"]

        # Courbe ROC pour le modèle
        ax.plot(fpr, tpr, color=colors[idx], label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')

        # Ajouter titres et légendes
        ax.set_title(f"{dataset_name} - {model_type}")
        ax.set_xlabel("False Positive Rate (FPR)")
        if idx == 0:
            ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend(loc='lower right')
        ax.grid(True)

    # Quatrième subplot : combiner toutes les courbes
    combined_ax = axes[-1]
    for idx, (model_type, data) in enumerate(model_results.items()):
        combined_ax.plot(data["fpr"], data["tpr"], color=colors[idx], label=f"{model_type}")
    combined_ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')
    combined_ax.set_title("All ROC Curves")
    combined_ax.set_xlabel("False Positive Rate (FPR)")
    combined_ax.legend(loc='lower right')
    combined_ax.grid(True)

    # Ajuster les espacements entre les subplots
    plt.tight_layout()
    plt.show()


def plot_accuracy_per_class_for_dataset(results, dataset_name):
    """
    Trace les graphiques de précision par classe pour chaque modèle associé à un dataset donné,
    avec l'accuracy globale affichée pour chaque modèle.

    Args:
        results (dict): Le dictionnaire contenant les résultats pour chaque dataset et modèle.
        dataset_name (str): Le nom du dataset à visualiser.
    """
    if dataset_name not in results:
        raise ValueError(f"Dataset {dataset_name} n'existe pas dans les résultats.")

    model_results = results[dataset_name]
    num_models = len(model_results)

    # Créer des subplots : 1 ligne avec autant de colonnes que de modèles
    fig, axes = plt.subplots(1, num_models, figsize=(num_models * 5, 5), sharey=True)

    # Si un seul modèle, convertir `axes` en une liste pour une manipulation cohérente
    if num_models == 1:
        axes = [axes]

    for idx, (model_type, data) in enumerate(model_results.items()):
        ax = axes[idx]

        # Récupérer les précisions par classe
        accuracy_per_class = data["accuracy_per_class"]

        # Récupérer l'accuracy globale
        global_accuracy = data["global_accuracy"]

        # Tracer les précisions sous forme de barres
        ax.bar(accuracy_per_class.keys(), accuracy_per_class.values(), color='skyblue')
        ax.set_title(f"{dataset_name} - {model_type}\nGlobal Accuracy: {global_accuracy:.2%}")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)  # Limite de l'axe Y pour toujours afficher de 0 à 100%
        ax.set_xticks(range(len(accuracy_per_class.keys())))
        ax.set_xticklabels(accuracy_per_class.keys(), rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)


    # Ajuster les espacements entre les subplots
    plt.tight_layout()
    plt.show()

def get_metrics_dataframe(results, dataset_name):
    """
    Retourne un DataFrame contenant les métriques des modèles associés à un dataset donné.
    
    Args:
        results (dict): Dictionnaire contenant les résultats.
        dataset_name (str): Le nom du dataset pour lequel les métriques doivent être extraites.

    Returns:
        pd.DataFrame: Un DataFrame avec les métriques des modèles associés au dataset.
    """
    if dataset_name not in results:
        raise ValueError(f"Dataset {dataset_name} n'existe pas dans les résultats.")

    # Extraire les résultats pour le dataset
    model_results = results[dataset_name]

    # Liste pour stocker les données sous forme de lignes
    rows = []

    for model_type, data in model_results.items():
        # Extraire les métriques
        roc_auc = data["roc_auc"]
        global_accuracy = data["global_accuracy"]
        precision = data["precision"]
        recall = data["recall"]
        f1_score = data["f1_score"]
        
        # Calculer TPR et FPR moyens
        tpr_mean = sum(data["tpr"]) / len(data["tpr"])  # Moyenne des TPR
        fpr_mean = sum(data["fpr"]) / len(data["fpr"])  # Moyenne des FPR
        
        # Ajouter une ligne pour le modèle
        rows.append({
            "Model": model_type,
            "AUROC": roc_auc,
            "Accuracy": global_accuracy,
            "TPR (mean)": tpr_mean,
            "FPR (mean)": fpr_mean,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
        })

    # Convertir les données en DataFrame
    df = pd.DataFrame(rows)

    return df

if __name__ == "__main__" :
    print(" todo va bene")
