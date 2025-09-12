from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import AbstractDetector

def evaluate_model(model, test_dataset):
    """Calcul de l'accuracy moyenne du modele sur le dataset de test
    """
    model.eval()

    all_predictions = []
    all_labels = []
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    #Transformation des probabilités de classification en choix binaire
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            labels = labels.float().unsqueeze(1)
            outputs = model(imgs)
            pred_batch = (outputs >= 0.5).float()
            all_predictions.append(pred_batch)
            all_labels.append(labels)

    #Concaténation des batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    #Calcul de l'accuracy moyenne
    accuracy = (all_predictions == all_labels).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")

if __name__ == "__main__":
    model = AbstractDetector()
    model.load_state_dict(torch.load("models/abstract_cnn.pth"))
    test_dataset = torch.load("models/test_dataset.pth")
    #evaluate_model(model, test_dataset)
    evaluate_on_directory(model, "/Users/john/Desktop/rdm/photos/imgs/*.jpg")