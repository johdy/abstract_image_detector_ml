import os
import numpy as np

import torch
import torchvision.transforms as transforms
import kagglehub
from torchvision import datasets
from torch.utils.data import ConcatDataset, Subset
import matplotlib.pyplot as plt


class FlattenDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label: int = 0):
        self.ds = dataset
        self.label = label
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        return x, self.label

def plot_dataset_ex(dataset, lines: int = 5):
    """Affichage du nombre voulu d'exemples du dataset
    """

    labels_map = {0: "photo", 1: "abstract"}
    figure = plt.figure(figsize=(8, 8))
    for i in range(1, lines * lines + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(lines, lines, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze().permute(1, 2, 0), cmap="gray")
    plt.show()

def clean_and_concat_datasets(photographs_dataset, abstract_dataset):
    """Création des subsets de tailles identiques et maximales, ensuite concaténés avec des labels binaires (0 ou 1)
    """

    #Récupération de la taille minimale des datasets pour uniformiser leur proportion
    min_dataset_len = min(len(photographs_dataset), len(abstract_dataset))

    #Shuffle, resize et applatissement des datasets
    photographs_indices = list(range(len(photographs_dataset)))
    np.random.shuffle(photographs_indices)
    photographs_subset = Subset(photographs_dataset, photographs_indices[:min_dataset_len])
    photographs_flat = FlattenDataset(photographs_subset, 0)

    paintings_indices = list(range(len(abstract_dataset)))
    np.random.shuffle(paintings_indices)
    paintings_subset = Subset(abstract_dataset, paintings_indices[:min_dataset_len])
    paintings_flat = FlattenDataset(paintings_subset, 1)
    
    #Concéténation des datasets
    dataset = ConcatDataset([photographs_flat, paintings_flat])
    plot_dataset_ex(dataset, 10)
    return dataset


def get_dataset(data_path: str):
    """Téléchargement et récupération du dataset clean et filtré
    """

    os.environ["KAGGLEHUB_CACHE"] = data_path

    #Création de la forme du transform
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    #Téléchargement des datasets de photos et peintures via kagglehub
    path_nature = kagglehub.dataset_download("pankajkumar2002/random-image-sample-dataset")
    path_paintings = kagglehub.dataset_download("steubk/wikiart")

    nature_dataset = datasets.ImageFolder(root=path_nature, transform=transform)
    paintings_dataset = datasets.ImageFolder(root=path_paintings, transform=transform)

    #Sélection des peintures abstraites dans le dataset
    abstract_classes = ["Abstract_Expressionism", "Action_painting", "Color_Field_Painting"]
    abstract_idx = [paintings_dataset.class_to_idx[idx] for idx in abstract_classes]
    abstract_samples = [sample for sample in paintings_dataset.samples if sample[1] in abstract_idx]
    paintings_dataset.samples = abstract_samples
    paintings_dataset.targets = [s[1] for s in abstract_samples]

    #Enrichissement du dataset de photo avec des visages humains
    path_people = kagglehub.dataset_download("ahmadahmadzada/images2000")
    people_dataset = datasets.ImageFolder(root=path_people, transform=transform)

    #Concaténation des photographies et des humains
    photographs_flat = FlattenDataset(nature_dataset, label=0)
    people_flat = FlattenDataset(people_dataset, label=1)
    photographs_dataset = ConcatDataset([photographs_flat, people_flat])
    print(len(photographs_dataset), len(paintings_dataset))
    return clean_and_concat_datasets(photographs_dataset, paintings_dataset)
