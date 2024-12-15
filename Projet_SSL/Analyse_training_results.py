import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import numpy as np
from collections import Counter
import pandas as pd
import warnings



file_path = 'training_results.csv'
data = pd.read_csv(file_path)

# Aperçu des premières lignes du fichier
#print(data.head())

# Extraire la dernière valeur de "Validation Loss" pour chaque configuration
data['Final Val Loss'] = data['Validation Loss'].apply(lambda x: eval(x)[-1])


top_1_per_config = (
    data.sort_values(by=['Model', 'Dataset', 'Final Val Loss'])
    .groupby(['Model', 'Dataset'])
    .head(1)
    .reset_index(drop=True)
)

#print(top_1_per_config)


# Tracer les courbes d'entraînement pour les top configurations
def plot_training_curves(top_configs, data):
    datasets = top_configs['Dataset'].unique()
    models = top_configs['Model'].unique()

    for dataset in datasets:
        # Filtrer les configurations pour le dataset courant
        dataset_configs = top_configs[top_configs['Dataset'] == dataset]
        
        # Initialiser la figure pour les subplots
        fig, axes = plt.subplots(1, len(models), figsize=(15, 5), sharey=True)
        fig.suptitle(f'Training Curves for Dataset: {dataset}', fontsize=16)

        for ax, model in zip(axes, models):
            # Filtrer pour le modèle courant
            model_config = dataset_configs[dataset_configs['Model'] == model]
            
            if not model_config.empty:
                # Extraire les paramètres et les courbes
                learning_rate = model_config['Learning Rate'].values[0]
                latent_dim = model_config['Latent Dim'].values[0]
                train_loss = eval(model_config['Train Loss'].values[0])
                val_loss = eval(model_config['Validation Loss'].values[0])

                # Tracer les courbes
                ax.plot(train_loss, label='Train Loss', color='blue')
                ax.plot(val_loss, label='Validation Loss', color='orange')
                ax.set_title(f'{model}\nLR: {learning_rate}, Latent Dim: {latent_dim}', fontsize=10)
                ax.set_xlabel('Epochs')
                ax.legend()
            else:
                ax.set_title(f'{model}\nNo data', fontsize=10)
            
            ax.set_ylabel('Loss')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # Ajuster les espacements
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Tracer les courbes d'entraînement pour les top 1 configurations
#plot_training_curves(top_1_per_config, data)



# Tracer les courbes d'entraînement pour les top configurations
def plot_training_curves_with_val_loss(top_configs, data):
    # Désactiver uniquement les UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    datasets = top_configs['Dataset'].unique()
    models = top_configs['Model'].unique()

    for dataset in datasets:
        # Filtrer les configurations pour le dataset courant
        dataset_configs = top_configs[top_configs['Dataset'] == dataset]
        
        # Initialiser la figure pour les subplots
        fig, axes = plt.subplots(1, len(models), figsize=(15, 5), sharey=True)
        fig.suptitle(f'Training Curves for Dataset: {dataset}', fontsize=16)

        for ax, model in zip(axes, models):
            # Filtrer pour le modèle courant
            model_config = dataset_configs[dataset_configs['Model'] == model]
            
            if not model_config.empty:
                # Extraire les paramètres et les courbes
                learning_rate = model_config['Learning Rate'].values[0]
                latent_dim = model_config['Latent Dim'].values[0]
                train_loss = eval(model_config['Train Loss'].values[0])
                val_loss = eval(model_config['Validation Loss'].values[0])
                final_val_loss = val_loss[-1]

                # Tracer les courbes
                ax.plot(train_loss, label='Train Loss', color='blue')
                ax.plot(val_loss, label='Validation Loss', color='orange')
                
                # Ligne horizontale pour la Validation Loss finale
                ax.axhline(final_val_loss, linestyle='--', color='red', alpha=0.7, label=f'Final Val Loss: {final_val_loss:.4f}')
                
                # Ajouter le titre
                ax.set_title(f'{model}\nLR: {learning_rate}, Latent Dim: {latent_dim}', fontsize=10)
                ax.set_xlabel('Epochs')
                ax.legend()
            else:
                ax.set_title(f'{model}\nNo data', fontsize=10)
            
            ax.set_ylabel('Loss')
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # Ajuster les espacements
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()






def plot_val_loss_with_colormap(data):
    models = data['Model'].unique()
    datasets = data['Dataset'].unique()

    for model in models:
        for dataset in datasets:
            # Filtrer les données pour le modèle et dataset courant
            subset = data[(data['Model'] == model) & (data['Dataset'] == dataset)]

            if not subset.empty:
                # Identifier le meilleur point (val loss la plus basse)
                best_point = subset.loc[subset['Final Val Loss'].idxmin()]
                best_val_loss = best_point['Final Val Loss']
                best_latent_dim = best_point['Latent Dim']
                best_learning_rate = best_point['Learning Rate']

                # Création du graphe
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(
                    subset['Latent Dim'], 
                    subset['Learning Rate'], 
                    c=subset['Final Val Loss'], 
                    cmap='viridis', 
                    s=100, 
                    edgecolor='k'
                )
                plt.colorbar(scatter, label='Final Validation Loss')

                # Mettre en évidence le meilleur point
                plt.scatter(
                    best_latent_dim, best_learning_rate, 
                    color='red', s=150, label=f'Best Point: {best_val_loss:.4f}'
                )
                plt.annotate(
                    f'{best_val_loss:.4f}', 
                    (best_latent_dim, best_learning_rate), 
                    textcoords="offset points", 
                    xytext=(10, 10), 
                    ha='center', color='red', fontsize=10
                )

                # Configurations du graphique
                plt.title(f'Model: {model} | Dataset: {dataset}', fontsize=14)
                plt.xlabel('Latent Dimension', fontsize=12)
                plt.ylabel('Learning Rate', fontsize=12)
                plt.xscale('log')
                plt.yscale('log')
                plt.legend()
                plt.grid(True, which="both", linestyle='--', linewidth=0.5)
                plt.show()

# Appeler la fonction pour tracer les graphes
#plot_val_loss_with_colormap(data)



def plot_val_loss_per_dataset_updated(data):
    datasets = data['Dataset'].unique()
    models = data['Model'].unique()

    for dataset in datasets:
        # Filtrer les données pour le dataset courant
        subset = data[data['Dataset'] == dataset]

        # Préparer les sous-graphes
        fig, axes = plt.subplots(1, len(models), figsize=(18, 6), sharey=True)
        fig.suptitle(f'Validation Loss for Dataset: {dataset}', fontsize=16)

        # Trouver toutes les valeurs uniques pour les ticks des axes
        latent_dims = sorted(subset['Latent Dim'].unique())
        learning_rates = sorted(subset['Learning Rate'].unique())

        for ax, model in zip(axes, models):
            # Filtrer pour le modèle courant
            model_data = subset[subset['Model'] == model]

            if not model_data.empty:
                # Identifier le meilleur point (val loss la plus basse)
                best_point = model_data.loc[model_data['Final Val Loss'].idxmin()]
                best_val_loss = best_point['Final Val Loss']
                best_latent_dim = best_point['Latent Dim']
                best_learning_rate = best_point['Learning Rate']

                # Création du scatter plot
                scatter = ax.scatter(
                    model_data['Latent Dim'], 
                    model_data['Learning Rate'], 
                    c=model_data['Final Val Loss'], 
                    cmap='viridis',  # Utilisation de l'ancien colormap
                    s=100, 
                    edgecolor='k'
                )
                # Mettre en évidence le meilleur point
                ax.scatter(
                    best_latent_dim, best_learning_rate, 
                    color='red', s=150, label=f'Best: {best_val_loss:.4f}'
                )
                ax.annotate(
                    f'{best_val_loss:.4f}', 
                    (best_latent_dim, best_learning_rate), 
                    textcoords="offset points", 
                    xytext=(10, -15), 
                    ha='center', color='red', fontsize=10
                )

                # Configurer les axes
                ax.set_title(f'Model: {model}', fontsize=12)
                ax.set_xlabel('Latent Dimension', fontsize=10)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xticks(latent_dims)
                ax.set_xticklabels([f'{int(ld)}' for ld in latent_dims], rotation=45)
                ax.set_yticks(learning_rates)
                ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])
                ax.grid(True, which="both", linestyle='--', linewidth=0.5)
                ax.legend()

        # Ajouter une barre colorée déplacée pour ne pas chevaucher les graphes
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position (droite des subplots)
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Final Validation Loss', fontsize=12)

        # Ajuster les espacements
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Réduire pour laisser la place à la barre
        plt.show()

# Appeler la fonction pour tracer les graphes
#plot_val_loss_per_dataset_updated(data)


def plot_val_loss_per_dataset_final(data):
    # Désactiver uniquement les UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
    datasets = data['Dataset'].unique()
    models = data['Model'].unique()

    for dataset in datasets:
        # Filtrer les données pour le dataset courant
        subset = data[data['Dataset'] == dataset]

        # Préparer les sous-graphes
        fig, axes = plt.subplots(1, len(models), figsize=(18, 6), sharey=True)
        fig.suptitle(f'Validation Loss for Dataset: {dataset}', fontsize=16)

        # Trouver toutes les valeurs uniques pour les ticks des axes
        latent_dims = sorted(subset['Latent Dim'].unique())
        learning_rates = sorted(subset['Learning Rate'].unique())

        for ax, model in zip(axes, models):
            # Filtrer pour le modèle courant
            model_data = subset[subset['Model'] == model]

            if not model_data.empty:
                # Identifier le meilleur point (val loss la plus basse)
                best_point = model_data.loc[model_data['Final Val Loss'].idxmin()]
                best_val_loss = best_point['Final Val Loss']
                best_latent_dim = best_point['Latent Dim']
                best_learning_rate = best_point['Learning Rate']

                # Création du scatter plot
                scatter = ax.scatter(
                    model_data['Latent Dim'], 
                    model_data['Learning Rate'], 
                    c=model_data['Final Val Loss'], 
                    cmap='viridis', 
                    s=100, 
                    edgecolor='k'
                )
                # Mettre en évidence le meilleur point
                ax.scatter(
                    best_latent_dim, best_learning_rate, 
                    color='red', s=150, label=f'Best: {best_val_loss:.4f}'
                )
                ax.annotate(
                    f'{best_val_loss:.4f}', 
                    (best_latent_dim, best_learning_rate), 
                    textcoords="offset points", 
                    xytext=(10, -15), 
                    ha='center', color='red', fontsize=10
                )

                # Configurer les axes
                ax.set_title(f'Model: {model}', fontsize=12)
                ax.set_xlabel('Latent Dimension', fontsize=10)

                # Définir les ticks uniquement avec les valeurs du CSV
                ax.set_xticks(latent_dims)
                ax.set_xticklabels([f'{int(ld)}' for ld in latent_dims], rotation=45)
                
                # Configurer l'axe des learning rates
                ax.set_yscale('log')
                ax.set_yticks(learning_rates)
                ax.set_yticklabels([f'{lr:.0e}' for lr in learning_rates])

                ax.grid(True, which="both", linestyle='--', linewidth=0.5)
                ax.legend()

        # Ajouter une barre colorée déplacée pour ne pas chevaucher les graphes
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position (droite des subplots)
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Final Validation Loss', fontsize=12)

        # Ajuster les espacements
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Réduire pour laisser la place à la barre
        plt.show()



if __name__ == "__main__":
	# Tracer les courbes d'entraînement pour les top 1 configurations
	plot_training_curves_with_val_loss(top_1_per_config, data)

	# Appeler la fonction pour tracer les graphes
	plot_val_loss_per_dataset_final(data)

