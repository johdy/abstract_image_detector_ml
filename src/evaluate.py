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

def evaluate_on_directory(model, path):
    from PIL import Image
    import torchvision.transforms as transforms

    # Ton transform habituel
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    from glob import glob
    import time
    # Charger l'image
    immagess = glob(path)
    #immagess = ["/Users/john/Desktop/0195377001666708189.jpg"]
    for ima in immagess:
        img_path = ima
        img = Image.open(img_path).convert("RGB")  # s'assure que c'est en 3 canaux

        # Appliquer le transform et ajouter une dimension batch
        img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 128, 128]

        # Mettre le modèle en mode evaluation
        model.eval()
        with torch.no_grad():  # pas de gradient nécessaire
            output = model(img_tensor)
            prob = output.item()  # sortie scalaire
            label = 1 if prob > 0.5 else 0

        print(f"Probabilité d'être une peinture: {prob:.3f}")
        print(f"Label prédit: {label}")
        img.show()
        time.sleep(4)


if __name__ == "__main__":
    model = AbstractDetector()
    model.load_state_dict(torch.load("models/abstract_cnn.pth"))
    test_dataset = torch.load("models/test_dataset.pth")
    #evaluate_model(model, test_dataset)
    evaluate_on_directory(model, "/Users/john/Desktop/rdm/photos/imgs/*.jpg")