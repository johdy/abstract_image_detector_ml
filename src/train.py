import os
from tqdm import tqdm
import json

import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

from model import AbstractDetector
from dataset import get_dataset, plot_dataset_ex

def train_model(path: str, batch_size: int = 32, lr: float = 0.001, epochs: int = 6):
    """Boucle d'entraînement du modèle, en fonction des hyperparamètres
    """
    dataset = get_dataset(path)
    dataset_size = len(dataset)

    #Division du dataset entre entraînement et test
    train_size = int(0.8 * dataset_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, dataset_size - train_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #Utilisation du GPU si possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Définition du modèle, de son critère de discrimination et de son optimiseur
    model = AbstractDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #Boucle d'entraînement
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    #Sauvegarde du modèle et du test_dataset, ainsi que des hyperparamètres
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/abstract_cnn.pth")
    torch.save(test_dataset, "models/test_dataset.pth")
    config = {"lr": lr, "batch_size": batch_size, "epochs": epochs, "img_size": (128,128)}
    with open("models/config.json", "w") as f:
        json.dump(config, f)

if __name__ == "__main__":
    train_model("/Volumes/John/wikiart")