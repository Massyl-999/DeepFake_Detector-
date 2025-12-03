# train.py - Entraînement uniquement sur images (dossiers real/ et fake/)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
from pathlib import Path

# === PARAMÈTRES ===
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = 128
MODEL_SAVE = "model.pth"

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === VÉRIFICATION DES DOSSIERS ===
real_dir = Path("real")
fake_dir = Path("fake")

if not real_dir.exists() or not fake_dir.exists():
    print("ERREUR : Dossiers 'real' et 'fake' introuvables !")
    print(f"   Créez les dossiers dans : {Path.cwd()}")
    exit()

# Chargement des données (seulement les dossiers real/ et fake/ à la racine)
try:
    dataset = datasets.ImageFolder(root=".", transform=transform_train)
    print(f"✓ Classes trouvées : {dataset.classes}")  # Doit afficher ['fake', 'real'] ou ['real', 'fake']
    print(f"✓ Total images : {len(dataset)}\n")
except Exception as e:
    print(f"Erreur : {e}")
    exit()

if len(dataset) == 0:
    print("ERREUR : Aucune image trouvée !")
    exit()

# Split 80% entraînement / 20% validation
val_split = 0.2
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
val_set.dataset.transform = transform_val

# === CHANGEMENT PRINCIPAL : num_workers=0 au lieu de 2 ===
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Modèle léger
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

best_acc = 0.0
print(f"Début de l'entraînement sur {len(dataset)} images ({device})\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train"):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch} | Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {acc:.4%}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"   >>> Modèle sauvegardé ! (Acc = {acc:.2%})")

print(f"\nEntraînement terminé ! Meilleur modèle : {MODEL_SAVE} (Accuracy = {best_acc:.2%})")