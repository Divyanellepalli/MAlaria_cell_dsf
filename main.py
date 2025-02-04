import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MalariaDataset
from mymodel import MosquitoNet

def generate_dataset(path):
    data = {'imgpath': [], 'labels': []}

    folders = os.listdir(path)

    for folder in folders:
        folderpath = os.path.join(path, folder)
        if os.path.isdir(folderpath):
            files = os.listdir(folderpath)
            for file in files:
                filepath = os.path.join(folderpath, file)
                if os.path.isfile(filepath):
                    data['imgpath'].append(filepath)
                    data['labels'].append(folder)

    return data

def main():
    # Paths and directories
    image_dir = "../input/cell-images-for-detecting-malaria/cell_images/"
    save_dir = "sample_images"
    model_path = "model.pth"

    # Hyperparameters
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transformations
    train_transforms = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Dataset
    dataset = MalariaDataset(image_dir=image_dir, transform=train_transforms)
    print("Length of the dataset:", len(dataset))

    # Model
    model = MosquitoNet().to(device)

    # DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_predictions * 100

        print(f"   Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
