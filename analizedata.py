import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the data directory
data_dir = '/kaggle/input/cell-images-for-detecting-malaria/cell_images'

# Function to generate the dataset
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

    return pd.DataFrame(data)

# Generate the dataset DataFrame
dataset = generate_dataset(data_dir)

# Define the function to plot and save samples
def plot_and_save_samples(df, num_pairs=4, save_dir='samples'):
    os.makedirs(save_dir, exist_ok=True)
    classes = df['labels'].unique()

    for i in range(num_pairs):
        plt.figure(figsize=(10, 5))

        sample_infected = df[df['labels'] == 'Parasitized'].sample(1)
        img_infected = cv2.imread(sample_infected.iloc[0]['imgpath'])
        img_infected = cv2.cvtColor(img_infected, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 1)
        plt.imshow(img_infected)
        plt.title('Infected')
        plt.axis('off')

        sample_uninfected = df[df['labels'] == 'Uninfected'].sample(1)
        img_uninfected = cv2.imread(sample_uninfected.iloc[0]['imgpath'])
        img_uninfected = cv2.cvtColor(img_uninfected, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 2, 2)
        plt.imshow(img_uninfected)
        plt.title('Uninfected')
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
        plt.close()

# Plot four pairs of infected and uninfected cell images and save them
plot_and_save_samples(dataset, num_pairs=4, save_dir='sample_images')
