import torch
import torch.nn as nn
from model import CardClassifier
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import  DataLoader
import os
import torch.optim as optim
dataset_dir="CardClassifier/dataset"
train_dir=os.path.join(dataset_dir,"train")
test_dir=os.path.join(dataset_dir,"test")
batch_size=16
num_epochs=15
lr=0.0005
num_classes=53
img_size=224
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device = ",device)
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
train_dataset=datasets.ImageFolder(train_dir,transform=train_transforms)
test_dataset=datasets.ImageFolder(test_dir,transform=test_transforms)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(test_dataset)}")
print(f"Classes found: {len(train_dataset.classes)}")
model=CardClassifier(num_classes=num_classes).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
train_losses, test_losses = [], []
train_accs, test_accs = [], []
best_val_acc = 0.0
print("\n Starting Training...\n")

for epoch in range(num_epochs):
    train_loss,train_acc=train_one_epoch(model,train_loader,optimizer, criterion)
    test_loss,test_acc=validate(model,test_loader,criterion)
    scheduler.step()

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.2f}%")
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✅ Best model saved! Val Acc: {test_acc:.2f}%")



plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.savefig("training_graph.png")
print("\nTraining complete! Graph saved as training_graph.png")

# Save class names for use in the web app
import json
with open("class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)
print("Class names saved to class_names.json")