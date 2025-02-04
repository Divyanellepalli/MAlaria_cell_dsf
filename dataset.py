import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define the MalariaDataset class
class MalariaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = os.listdir(image_dir)

        # Iterate over classes (subdirectories)
        for i, cls in enumerate(self.classes):
            class_dir = os.path.join(image_dir, cls)
            # Iterate over images in each class directory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path) and img_name.endswith(('.png', '.jpg', '.jpeg')):  # Check if file is an image
                    self.images.append(img_path)
                    self.labels.append(i)  # Assign label based on class index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Read the image
        try:
            image = cv2.imread(img_path)
            if image is None:  # Check if image could not be read
                raise Exception("Image could not be read")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = Image.fromarray(image)  # Convert to PIL Image
        except Exception as e:
            print(f"Error reading image at path: {img_path}")
            print(f"Error message: {e}")
            # Return a placeholder image if image cannot be read
            image = Image.new('RGB', (120, 120), (255, 255, 255))  # White image
            label = -1  # Assign a dummy label

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
test_transforms = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
])

# Define the image directory
image_dir = "../input/cell-images-for-detecting-malaria/cell_images/"

# Create an instance of the dataset
dataset = MalariaDataset(image_dir=image_dir, transform=test_transforms)

# Test length of the dataset
print("Length of the dataset:", len(dataset))

# Test the __getitem__ method
for i in range(5):  # Test with 5 samples
    # Select images from different classes
    idx_label_0 = dataset.labels.index(0)
    idx_label_1 = dataset.labels.index(1)

    image_label_0, label_0 = dataset[idx_label_0 + i]  # Images with label 0
    image_label_1, label_1 = dataset[idx_label_1 + i]  # Images with label 1

    # Convert tensor to numpy array and transpose dimensions for plotting
    image_label_0 = image_label_0.numpy().transpose((1, 2, 0))
    image_label_1 = image_label_1.numpy().transpose((1, 2, 0))

    # Plot the images
    plt.subplot(2, 5, i + 1)
    plt.imshow(image_label_0)
    plt.title(f"Label: {label_0}")

    plt.subplot(2, 5, i + 6)
    plt.imshow(image_label_1)
    plt.title(f"Label: {label_1}")

plt.tight_layout()
plt.show()

# Test integration over classes
class_counts = {cls: 0 for cls in dataset.classes}
for _, label in dataset:
    if label != -1:  # Skip dummy labels
        class_counts[dataset.classes[label]] += 1

print("Class counts:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")

