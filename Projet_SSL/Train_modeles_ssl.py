import pandas as pd
import torch
import torch.nn as nn
import os
from Load_dataset import (
    load_autoVI, load_bottle, load_capsule, load_hazelnut, load_toothbrush
)
from Modeles_code_training import MaskedAutoencoderModel, ColorizationModel, InpaintingModel, train_ssl_model
from torch.optim import Adam

# Initialisation de l'appareil (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


AutoVI_train_dataset, AutoVI_test_dataset, AutoVI_train_loader, AutoVI_test_loader, filtered_AutoVI_test_dataset, filtered_AutoVI_test_loader = load_autoVI()
bottle_train_dataset, bottle_test_dataset, bottle_train_loader, bottle_test_loader, filtered_bottle_test_dataset, filtered_bottle_test_loader = load_bottle()
toothbrush_train_dataset, toothbrush_test_dataset, toothbrush_train_loader, toothbrush_test_loader, filtered_toothbrush_test_dataset, filtered_toothbrush_test_loader = load_toothbrush()
hazelnut_train_dataset, hazelnut_test_dataset, hazelnut_train_loader, hazelnut_test_loader, filtered_hazelnut_test_dataset, filtered_hazelnut_test_loader = load_hazelnut()
capsule_train_dataset, capsule_test_dataset, capsule_train_loader, capsule_test_loader, filtered_capsule_test_dataset, filtered_capsule_test_loader = load_capsule()

# ------------------------------------------------------------- 
#                            Bottle
# ------------------------------------------------------------- 

# Auto Encoder :
epoch = 40
mae_model_bottle = MaskedAutoencoderModel(latent_dim=512)

# Entraîner le modèle et récupérer les pertes
mae_encoder_bottle, bottle_train_losses, bottle_val_losses = train_ssl_model(
    mae_model_bottle,
    bottle_train_loader, 
    filtered_bottle_test_loader, 
    criterion=nn.MSELoss(), 
    optimizer=Adam(mae_model_bottle.parameters(), lr=0.001),
    epochs=epoch
)

torch.save(mae_model_bottle.state_dict(), "Modeles/mae_model_bottle.pth")
print("Modèle sauvegardé sous le nom 'mae_model_bottle.pth'")

# Colorisation :

epoch = 40 #80
colorization_model_bottle = ColorizationModel(latent_dim=64)
colorization_encoder_bottle , bottle_train_losses, bottle_val_losses = train_ssl_model(colorization_model_bottle,
                              bottle_train_loader, 
                              filtered_bottle_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(colorization_model_bottle.parameters(), lr=0.01),
                              epochs=epoch
                              )
torch.save(colorization_model_bottle.state_dict(), "Modeles/colorization_model_bottle.pth")
print("Modèle sauvegardé sous le nom 'colorization_model_bottle.pth'")

# Inpainting :

epoch = 40 #80
Inpainting_model_bottle = InpaintingModel(latent_dim=128)
Inpainting_encoder_bottle , bottle_train_losses, bottle_val_losses = train_ssl_model(Inpainting_model_bottle,
                              bottle_train_loader, 
                              filtered_bottle_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(Inpainting_model_bottle.parameters(), lr=0.001),
                              epochs=epoch
                              )

torch.save(Inpainting_model_bottle.state_dict(), "Modeles/Inpainting_model_bottle.pth")
print("Modèle sauvegardé sous le nom 'Inpainting_model_bottle.pth'")

print(" ")
print("Entrainement sur Bottle terminé ! ")

# ------------------------------------------------------------- 
#                            AutoVI
# ------------------------------------------------------------- 

# Auto Encoder :
epoch = 40
mae_model_AutoVI = MaskedAutoencoderModel(latent_dim=512)

# Entraîner le modèle et récupérer les pertes
mae_encoder_AutoVI, AutoVI_train_losses, AutoVI_val_losses = train_ssl_model(
    mae_model_AutoVI,
    AutoVI_train_loader, 
    filtered_AutoVI_test_loader, 
    criterion=nn.MSELoss(), 
    optimizer=Adam(mae_model_AutoVI.parameters(), lr=0.001),
    epochs=epoch
)

torch.save(mae_model_AutoVI.state_dict(), "Modeles/mae_model_AutoVI.pth")
print("Modèle sauvegardé sous le nom 'mae_model_AutoVI.pth'")

# Colorisation :

epoch = 40
colorization_model_AutoVI = ColorizationModel(latent_dim=128)
colorization_encoder_AutoVI, AutoVI_train_losses, AutoVI_val_losses = train_ssl_model(colorization_model_AutoVI,
                              AutoVI_train_loader, 
                              filtered_AutoVI_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(colorization_model_AutoVI.parameters(), lr=0.001),
                              epochs=epoch
                              )
torch.save(colorization_model_AutoVI.state_dict(), "Modeles/colorization_model_AutoVI.pth")
print("Modèle sauvegardé sous le nom 'colorization_model_AutoVI.pth'")

# Inpainting :

epoch = 40
Inpainting_model_AutoVI = InpaintingModel(latent_dim=256)
Inpainting_encoder_AutoVI, AutoVI_train_losses, AutoVI_val_losses = train_ssl_model(Inpainting_model_AutoVI,
                              AutoVI_train_loader, 
                              filtered_AutoVI_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(Inpainting_model_AutoVI.parameters(), lr=0.001),
                              epochs=epoch
                              )

torch.save(Inpainting_model_AutoVI.state_dict(), "Modeles/Inpainting_model_AutoVI.pth")
print("Modèle sauvegardé sous le nom 'Inpainting_model_AutoVI.pth'")


print(" ")
print("Entrainement sur AutoVI terminé ! ")

# ------------------------------------------------------------- 
#                           Hazelnut
# ------------------------------------------------------------- 

# Auto Encoder :
epoch = 40
mae_model_hazelnut = MaskedAutoencoderModel(latent_dim=256)

# Entraîner le modèle et récupérer les pertes
mae_encoder_hazelnut, hazelnut_train_losses, hazelnut_val_losses = train_ssl_model(
    mae_model_hazelnut,
    hazelnut_train_loader, 
    filtered_hazelnut_test_loader, 
    criterion=nn.MSELoss(), 
    optimizer=Adam(mae_model_hazelnut.parameters(), lr=0.001),
    epochs=epoch
)

torch.save(mae_model_hazelnut.state_dict(), "Modeles/mae_model_hazelnut.pth")
print("Modèle sauvegardé sous le nom 'mae_model_hazelnut.pth'")

# Colorisation :

epoch = 40
colorization_model_hazelnut = ColorizationModel(latent_dim=512)
colorization_encoder_hazelnut, hazelnut_train_losses, hazelnut_val_losses = train_ssl_model(colorization_model_hazelnut,
                              hazelnut_train_loader, 
                              filtered_hazelnut_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(colorization_model_hazelnut.parameters(), lr=0.001),
                              epochs=epoch
                              )
torch.save(colorization_model_hazelnut.state_dict(), "Modeles/colorization_model_hazelnut.pth")
print("Modèle sauvegardé sous le nom 'colorization_model_hazelnut.pth'")

# Inpainting :

epoch = 40
Inpainting_model_hazelnut = InpaintingModel(latent_dim=128)
Inpainting_encoder_hazelnut, hazelnut_train_losses, hazelnut_val_losses = train_ssl_model(Inpainting_model_hazelnut,
                              hazelnut_train_loader, 
                              filtered_hazelnut_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(Inpainting_model_hazelnut.parameters(), lr=0.001),
                              epochs=epoch
                              )

torch.save(Inpainting_model_hazelnut.state_dict(), "Modeles/Inpainting_model_hazelnut.pth")
print("Modèle sauvegardé sous le nom 'Inpainting_model_hazelnut.pth'")

print(" ")
print("Entrainement sur Hazelnut terminé ! ")

# ------------------------------------------------------------- 
#                           Toothbrush
# ------------------------------------------------------------- 

# Auto Encoder :
epoch = 45
mae_model_toothbrush = MaskedAutoencoderModel(latent_dim=64)

# Entraîner le modèle et récupérer les pertes
mae_encoder_toothbrush, toothbrush_train_losses, toothbrush_val_losses = train_ssl_model(
    mae_model_toothbrush,
    toothbrush_train_loader, 
    filtered_toothbrush_test_loader, 
    criterion=nn.MSELoss(), 
    optimizer=Adam(mae_model_toothbrush.parameters(), lr=0.001),
    epochs=epoch
)

torch.save(mae_model_toothbrush.state_dict(), "Modeles/mae_model_toothbrush.pth")
print("Modèle sauvegardé sous le nom 'mae_model_toothbrush.pth'")

# Colorisation :

epoch = 45
colorization_model_toothbrush = ColorizationModel(latent_dim=256)
colorization_encoder_toothbrush, toothbrush_train_losses, toothbrush_val_losses = train_ssl_model(colorization_model_toothbrush,
                              toothbrush_train_loader, 
                              filtered_toothbrush_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(colorization_model_toothbrush.parameters(), lr=0.001),
                              epochs=epoch
                              )
torch.save(colorization_model_toothbrush.state_dict(), "Modeles/colorization_model_toothbrush.pth")
print("Modèle sauvegardé sous le nom 'colorization_model_toothbrush.pth'")

# Inpainting :

epoch = 45
Inpainting_model_toothbrush = InpaintingModel(latent_dim=64)
Inpainting_encoder_toothbrush, toothbrush_train_losses, toothbrush_val_losses = train_ssl_model(Inpainting_model_toothbrush,
                              toothbrush_train_loader, 
                              filtered_toothbrush_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(Inpainting_model_toothbrush.parameters(), lr=0.01),
                              epochs=epoch
                              )

torch.save(Inpainting_model_toothbrush.state_dict(), "Modeles/Inpainting_model_toothbrush.pth")
print("Modèle sauvegardé sous le nom 'Inpainting_model_toothbrush.pth'")

print(" ")
print("Entrainement sur Toothbrush terminé ! ")

# ------------------------------------------------------------- 
#                            Capsule
# ------------------------------------------------------------- 

# Auto Encoder :
epoch = 40
mae_model_capsule = MaskedAutoencoderModel(latent_dim=256)

# Entraîner le modèle et récupérer les pertes
mae_encoder_capsule, capsule_train_losses, capsule_val_losses = train_ssl_model(
    mae_model_capsule,
    capsule_train_loader, 
    filtered_capsule_test_loader, 
    criterion=nn.MSELoss(), 
    optimizer=Adam(mae_model_capsule.parameters(), lr=0.001),
    epochs=epoch
)

torch.save(mae_model_capsule.state_dict(), "Modeles/mae_model_capsule.pth")
print("Modèle sauvegardé sous le nom 'mae_model_capsule.pth'")

# Colorisation :

epoch = 40
colorization_model_capsule = ColorizationModel(latent_dim=128)
colorization_encoder_capsule, capsule_train_losses, capsule_val_losses = train_ssl_model(colorization_model_capsule,
                              capsule_train_loader, 
                              filtered_capsule_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(colorization_model_capsule.parameters(), lr=0.001),
                              epochs=epoch
                              )
torch.save(colorization_model_capsule.state_dict(), "Modeles/colorization_model_capsule.pth")
print("Modèle sauvegardé sous le nom 'colorization_model_capsule.pth'")

# Inpainting :

epoch = 40
Inpainting_model_capsule = InpaintingModel(latent_dim=256)
Inpainting_encoder_capsule, capsule_train_losses, capsule_val_losses = train_ssl_model(Inpainting_model_capsule,
                              capsule_train_loader, 
                              filtered_capsule_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=Adam(Inpainting_model_capsule.parameters(), lr=0.001),
                              epochs=epoch
                              )

torch.save(Inpainting_model_capsule.state_dict(), "Modeles/Inpainting_model_capsule.pth")
print("Modèle sauvegardé sous le nom 'Inpainting_model_capsule.pth'")

     
print(" ")
print("Entrainement sur Capsule terminé ! ")
