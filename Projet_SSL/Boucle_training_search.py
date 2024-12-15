import pandas as pd
import torch
import os
from Load_dataset import (
    load_autoVI, load_bottle, load_capsule, load_hazelnut, load_toothbrush
)
from Modeles_code_training import MaskedAutoencoderModel, ColorizationModel, InpaintingModel, train_ssl_model
from torch.optim import Adam

# Initialisation de l'appareil (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fonction pour entraîner un modèle et enregistrer les résultats
def train_and_log_results(model, dataset_name, train_loader, test_loader, lr, latent_dim, epochs, log_interval,model_name):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()  # Exemple, ajuster en fonction du modèle
    model, train_losses, val_losses = train_ssl_model(model = model, train_loader = train_loader, test_loader = test_loader, optimizer = optimizer, criterion = criterion, device = device, epochs = epochs)
    #val_loss, val_accuracy = evaluate_model(model, test_loader, criterion, device)
    results= ({
        'Model': model_name ,#model.__class__.__name__,
        'Dataset': dataset_name,
        'Latent Dim': latent_dim,
        'Learning Rate': lr,
        'Epoch': epochs,
        'Validation Loss': val_losses,
        'Train Loss': train_losses
    })

    return results

# Chargement des datasets
datasets_loaders = {
    "AutoVI": load_autoVI(),
    "Bottle": load_bottle(),
    "Capsule": load_capsule(),
    "Hazelnut": load_hazelnut(),
    "Toothbrush": load_toothbrush()
}

# Paramètres d'entraînement
learning_rates = [0.0001, 0.001, 0.05, 0.01]
latent_dims = [64, 128, 256, 512]
epochs = 40 #40
log_interval = 10 #10
models = [MaskedAutoencoderModel, ColorizationModel, InpaintingModel]

# Initialiser un DataFrame pour enregistrer les résultats
results_df = pd.DataFrame()


# Entraîner chaque modèle avec chaque dataset et combinaisons de paramètres
for model_class in models:
    print(model_class)
    for dataset_name, (train_dataset, _, train_loader, _, test_dataset, test_loader) in datasets_loaders.items():
        for lr in learning_rates:
            for latent_dim in latent_dims:
                model_name = model_class.__name__
                print(f"Training {model_class.__name__} on {dataset_name} with LR={lr} and Latent Dim={latent_dim}")

                # Initialiser le modèle
                model = model_class(latent_dim=latent_dim).to(device)

                # Entraîner et récupérer les résultats
                results = train_and_log_results(
                    model, dataset_name, train_loader, test_loader, lr, latent_dim, epochs, log_interval,model_name
                )

                # Ajouter les résultats au DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)


# Sauvegarder les résultats dans un fichier CSV
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/training_results.csv", index=False)
print("Training complete! Results saved to 'results/training_results.csv'")
