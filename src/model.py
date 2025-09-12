import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractDetector(nn.Module):
    def __init__(self):
        super().__init__()
        #Premier layer convolutionnel
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        #Second layer convolutionnel, de plus haut niveau
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        #Layers de connections linéaires
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

        #Donwsampling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        #Premiers layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        #Aplatissement du tensor
        x = x.view(-1, 64 * 32 * 32)

        #Layers linéaires pour classification finale
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x