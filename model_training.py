# ===============================
# 0️⃣ Imports
# ===============================
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
import copy
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1️⃣ Dataset Preparation
# ===============================
# Path to dataset
data_dir = "/root/.cache/kagglehub/datasets/thedagger/pokemon-generation-one/versions/1/dataset"

# Get classes and counts
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
class_counts = {}
for cls in classes:
    class_path = os.path.join(data_dir, cls)
    num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if num_images > 0:
        class_counts[cls] = num_images

valid_class_names = list(class_counts.keys())
print(f"Number of non-empty classes: {len(valid_class_names)}")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Filter out empty classes
filtered_samples = [(fp, idx) for fp, idx in full_dataset.samples if full_dataset.classes[idx] in valid_class_names]
new_class_to_idx = {cls:i for i, cls in enumerate(valid_class_names)}
full_dataset.samples = [(fp, new_class_to_idx[full_dataset.classes[idx]]) for fp, idx in filtered_samples]
full_dataset.targets = [s[1] for s in full_dataset.samples]
full_dataset.classes = valid_class_names
full_dataset.class_to_idx = new_class_to_idx

print(f"Number of classes after filtering: {len(full_dataset.classes)}")
print(f"Total images after filtering: {len(full_dataset)}")

# Split dataset
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
test_dataset.dataset.transform = test_transform

# Weighted sampler for skewed classes
train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
counts_array = np.array([class_counts[cls] for cls in full_dataset.classes])
samples_weight = [1.0 / counts_array[t] for t in train_targets]
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# ===============================
# 2️⃣ Model Definition
# ===============================
class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3))
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.4))
        self.conv5 = nn.Sequential(nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5))
        self.fc1 = nn.Linear(512*4*4,1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512,num_classes)
    
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x); x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCNN(num_classes=len(full_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(model)

# ===============================
# 3️⃣ Training Function
# ===============================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*30)
        
        # Training
        model.train()
        running_loss, all_preds, all_labels = 0.0, [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "alpha.h5")
            print("Best model updated!")

    model.load_state_dict(best_model_wts)
    return model, history

# ===============================
# 4️⃣ Train Model
# ===============================
trained_model, history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, device=device)
torch.save(trained_model.state_dict(), "alpha.h5")
print("Final model saved as alpha.h5")

# ===============================
# 5️⃣ Plot Training Curves
# ===============================
plt.figure(figsize=(10,5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Loss vs Epochs"); plt.legend(); plt.show()

plt.figure(figsize=(10,5))
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Val Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epochs"); plt.legend(); plt.show()

plt.figure(figsize=(10,5))
plt.plot(history["val_f1"], label="Validation F1-score")
plt.xlabel("Epochs"); plt.ylabel("F1-score"); plt.title("Validation F1-score vs Epochs"); plt.legend(); plt.show()

# ===============================
# 6️⃣ Test Predictions
# ===============================
trained_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = trained_model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_acc = accuracy_score(all_labels, all_preds)
final_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Final Test Accuracy: {final_acc:.4f}")
print(f"Final Test Macro F1-score: {final_f1:.4f}")

# Show example predictions
import matplotlib.pyplot as plt
trained_model.eval()
test_iter = iter(test_loader)
images, labels = next(test_iter)
images, labels = images.to(device), labels.to(device)
outputs = trained_model(images)
preds = torch.argmax(outputs, dim=1)
images = images.cpu().permute(0,2,3,1).numpy()

plt.figure(figsize=(15,6))
for i in range(min(8, len(images))):
    plt.subplot(2,4,i+1)
    plt.imshow((images[i]*0.5 + 0.5))  # denormalize
    plt.title(f"Pred: {full_dataset.classes[preds[i]]}\nLabel: {full_dataset.classes[labels[i]]}")
    plt.axis('off')
plt.show()