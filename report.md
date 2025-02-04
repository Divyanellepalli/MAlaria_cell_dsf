# Malaria Cell Image Classification

---

## Introduction:

Welcome to the Malaria Cell Image Classification project, a groundbreaking endeavor aimed at leveraging cutting-edge technology to combat malaria, a life-threatening disease that affects millions of people worldwide. In this project, we focus on utilizing machine learning techniques to develop an automated system capable of accurately classifying microscopic images of blood cells as either infected or uninfected with the malaria parasite.

The dataset used in this project is sourced from the official NIH website and comprises two main categories: "Infected" and "Uninfected". With a total of 27,558 images, this dataset provides a rich and diverse collection of malaria-infected and healthy blood cell images, enabling us to train and evaluate our machine learning models effectively.

Our primary objective is to develop a robust classification model that can accurately distinguish between malaria-infected and uninfected blood cells. By achieving this goal, we aim to streamline the process of malaria diagnosis, particularly in resource-constrained settings where access to skilled healthcare professionals may be limited.

Through this project, we seek to harness the power of technology to make a meaningful impact in the fight against malaria. By automating the diagnosis process, we can expedite treatment and improve health outcomes for individuals affected by this devastating disease.

## Samples Visualization: 

The process of generating a dataset from a directory containing images of malaria-infected and uninfected cells. It then plots and saves four pairs of infected and uninfected cell images for visualization purposes.

Here's a breakdown of how this task was approached:

1. **Data Directory Setup**: The first step is to define the path to the data directory containing the images. In this case, the directory is '/kaggle/input/cell-images-for-detecting-malaria/cell_images'.

2. **Dataset Generation Function**: A function named `generate_dataset` is defined to traverse through the data directory and create a DataFrame containing the image paths and corresponding labels (infected or uninfected). This function iterates over the subdirectories within the main data directory, reads each image file, and appends its path and label to the DataFrame.

3. **DataFrame Creation**: The `generate_dataset` function is called to create a DataFrame named `dataset`, which contains the image paths and labels.

4. **Plotting and Saving Samples Function**: Another function named `plot_and_save_samples` is defined to visualize and save sample pairs of infected and uninfected cell images. This function randomly samples one infected and one uninfected image from the dataset DataFrame, reads and plots them using Matplotlib, and then saves the plot as a PNG file. This process is repeated for the specified number of pairs.

5. **Visualization and Saving**: Finally, the `plot_and_save_samples` function is called to generate four pairs of infected and uninfected cell images and save them in a directory named 'sample_images'.

This approach allows for the efficient generation of a dataset DataFrame and the visualization of sample images, enabling quick inspection of the data and verification of its integrity before proceeding with further analysis or model training.

First Sample:

![first_image](https://github.com/Ashishlathkar77/Malaria-Cell-Image-Classification/blob/main/Visualizations/sample_1.png)

Second Sample:

![first_image](https://github.com/Ashishlathkar77/Malaria-Cell-Image-Classification/blob/main/Visualizations/sample_2.png)

Third Sample:

![first_image](https://github.com/Ashishlathkar77/Malaria-Cell-Image-Classification/blob/main/Visualizations/sample_3.png)

Fourth Sample:

![first_image](https://github.com/Ashishlathkar77/Malaria-Cell-Image-Classification/blob/main/Visualizations/sample_4.png)

## MalariaDataset and Transformations:

Defines a custom dataset class called `MalariaDataset` for handling malaria cell images. Additionally, it includes preprocessing and transformation steps using PyTorch's `transforms` module.

Here's how this approach was implemented:

1. **Custom Dataset Class (MalariaDataset)**:
   - The `MalariaDataset` class is created by subclassing `torch.utils.data.Dataset`, making it compatible with PyTorch's data loading utilities.
   - It initializes with parameters such as the directory containing the image data (`image_dir`) and any desired transformations (`transform`).
   - In the constructor (`__init__` method), it populates lists of image paths (`self.images`) and corresponding labels (`self.labels`). Labels are assigned based on the subdirectory names within the `image_dir`.
   - The `__len__` method returns the total number of images in the dataset.
   - The `__getitem__` method retrieves an image and its corresponding label at a given index. It reads the image using OpenCV, converts it to RGB format, and applies any specified transformations.
   - Exception handling is implemented to handle cases where images cannot be read. In such cases, a placeholder image is returned with a dummy label.

2. **Preprocessing and Transformation**:
   - Transformation operations are defined using PyTorch's `transforms.Compose` class, allowing for sequential application of transformations.
   - Two transformations are specified: resizing images to a fixed size of 120x120 pixels and converting images to tensors using `transforms.ToTensor()`.
   - These transformations are then applied to the images in the dataset during training or inference.

3. **Data Loading and Visualization**:
   - The dataset is instantiated (`dataset = MalariaDataset(image_dir=image_dir, transform=test_transforms)`), providing the directory containing the malaria cell images and the defined transformations.
   - The length of the dataset is printed to verify the number of images.
   - Five sample pairs of images are selected from different classes (infected and uninfected) for visualization. These images are converted to NumPy arrays and plotted using Matplotlib.
   - Class counts are calculated to determine the distribution of images across different classes in the dataset.

This approach provides a structured and scalable solution for handling image data, including loading, preprocessing, and transformation, making it suitable for training deep learning models for malaria cell image classification.

## MosquitoNet - MyModel:

Defines a convolutional neural network (CNN) model called `MosquitoNet` using PyTorch's `nn.Module` class for classifying malaria cell images. Here's how we can explain the approach:

1. **Model Architecture**:
   - The `MosquitoNet` model consists of three convolutional layers (`layer1`, `layer2`, and `layer3`) followed by fully connected layers (`fc1`, `fc2`, and `fc3`).
   - Each convolutional layer is defined as a sequential combination of operations:
     - Convolutional operation using `nn.Conv2d` to extract features from input images.
     - Batch normalization (`nn.BatchNorm2d`) to improve the stability and speed of training by normalizing the input to each layer.
     - ReLU activation function (`nn.ReLU`) to introduce non-linearity.
     - Max pooling (`nn.MaxPool2d`) to downsample the feature maps and reduce spatial dimensions.
   - The fully connected layers (`fc1`, `fc2`, and `fc3`) are responsible for combining the features extracted by convolutional layers and making predictions.
   - Dropout (`nn.Dropout2d`) is applied to the fully connected layers to prevent overfitting by randomly setting a fraction of input units to zero during training.

2. **Forward Method**:
   - The `forward` method defines the forward pass of the model.
   - Input images (`x`) are passed through each layer sequentially, applying the defined operations.
   - The output of the final fully connected layer (`fc3`) represents the class scores for each image.
   - The model returns the output scores, which can be used to compute loss and make predictions.

3. **Device Specification**:
   - The code checks whether a GPU (CUDA) is available using `torch.cuda.is_available()`.
   - If a GPU is available, the model is moved to the GPU device (`cuda:0`) using `model.to(device)`.
   - If no GPU is available, the model remains on the CPU device.

4. **Model Instantiation**:
   - An instance of the `MosquitoNet` model is created using `model = MosquitoNet()`.
   - The model is then moved to the appropriate device (GPU or CPU) using `model.to(device)`.

Overall, this approach follows a standard CNN architecture for image classification tasks, incorporating convolutional layers for feature extraction and fully connected layers for classification. Dropout regularization is used to prevent overfitting, and the model is dynamically moved to the available hardware device for efficient computation.

![fourth_image](https://github.com/Ashishlathkar77/Malaria-Cell-Image-Classification/blob/main/Visualizations/Screenshot%20(2346).png)

## Training and Testing:

The `train.py` script is responsible for training a deep learning model for malaria cell image classification using PyTorch. Here's an explanation of how the approach is structured:

1. **Importing Libraries**:
   - The script starts by importing necessary libraries such as `torch`, `torch.nn`, `torch.optim`, and `torch.utils.data` for building and training neural networks, as well as `transforms` from `torchvision` for image transformations.

2. **Defining Device and Hyperparameters**:
   - The device (GPU or CPU) is defined based on the availability of CUDA.
   - Hyperparameters such as the number of epochs, batch size, and learning rate are specified.

3. **Defining Data Transformations**:
   - Data transformations are defined using `transforms.Compose`. These transformations include resizing, random horizontal and vertical flips, random rotation, converting images to tensors, and normalization.

4. **Loading Dataset and Data Loader**:
   - The `MalariaDataset` class from `dataset.py` is used to load the dataset. The dataset directory and defined transformations are passed as arguments.
   - A `DataLoader` is created to iterate over batches of the dataset during training.

5. **Instantiating Model**:
   - An instance of the `MosquitoNet` model is created from `mymodel.py`. The model is moved to the specified device.

6. **Defining Loss Function and Optimizer**:
   - The cross-entropy loss function and the Adam optimizer are defined.
   - The model parameters are passed to the optimizer for optimization.

7. **Training Loop**:
   - The script enters a loop over the specified number of epochs.
   - Within each epoch, the model is set to training mode (`model.train()`).
   - The training data loader is iterated over, and for each batch:
     - Input images and labels are moved to the specified device.
     - The optimizer gradients are zeroed.
     - Forward pass is performed through the model, and loss is calculated.
     - Backward pass is executed, and model parameters are updated.
     - Training loss, accuracy, and other statistics are updated.
   - At the end of each epoch, the average loss and accuracy for the epoch are printed.

8. **Saving the Trained Model**:
   - After training, the trained model's state dictionary is saved to a file named `model.pth` using `torch.save`.

Overall, this approach follows a standard training procedure for deep learning models, including data loading, model instantiation, defining loss and optimizer, training loop, and model saving. The script is structured to ensure efficient training and easy reproducibility.

## Main Script for Project:

In this project, we approached the task of classifying malaria cell images into infected and uninfected categories using a systematic and modular approach. Here's how we approached each component:

1. **Dataset Generation (`dataset.py`):**
   - We began by creating a custom dataset class named `MalariaDataset`.
   - This class reads the image files from the specified directory and generates a dataset containing image paths and corresponding labels (infected or uninfected).
   - We handled data loading, preprocessing, and labeling within this class to ensure compatibility with PyTorch's `DataLoader`.

2. **Model Definition (`mymodel.py`):**
   - Next, we defined the architecture of the neural network model responsible for image classification.
   - Our model, named `MosquitoNet`, is a convolutional neural network (CNN) consisting of multiple layers, including convolutional, batch normalization, activation, max-pooling, and fully connected layers.
   - We used PyTorch's `nn.Module` to create a modular and customizable model architecture.

3. **Training (`train.py`):**
   - We then implemented the training pipeline to train the model on the generated dataset.
   - This involved setting up hyperparameters such as the number of epochs, batch size, and learning rate.
   - We defined data transformations, including resizing, random horizontal and vertical flips, random rotation, normalization, to augment the dataset and improve model generalization.
   - The training loop iterated over the dataset batches, performed forward and backward passes, updated model parameters using gradient descent, and calculated training loss and accuracy.
   - We saved the trained model parameters to disk for future use or inference.

4. **Integration (`main.py`):**
   - Finally, we integrated the dataset generation, model definition, and training pipeline into a single `main.py` script.
   - This script orchestrates the entire process, from dataset generation to model training, in a modular and organized manner.
   - By encapsulating each component into separate files (`dataset.py`, `mymodel.py`, `train.py`), we promote code reusability, readability, and maintainability.

Overall, our approach emphasizes modularity, scalability, and flexibility, allowing for easy experimentation, customization, and extension of the malaria cell image classification pipeline. We leverage the power of PyTorch and its ecosystem to streamline the development process and achieve accurate and reliable results.

## Results:

The results indicate the performance of our malaria cell image classification model on the test set. Let's break down the key metrics and findings:

1. **Overall Accuracy:**
   - The overall accuracy on the test set is reported as 96.13%. This indicates that our model correctly classified 96.13% of the images into their respective categories (infected or uninfected).

2. **Accuracy by Class:**
   - The accuracy for classifying infected images is reported as 94.93%. This means that among all the infected images in the test set, 94.93% were correctly classified by our model.
   - The accuracy for classifying uninfected images is reported as 97.36%. Similarly, among all the uninfected images in the test set, 97.36% were correctly classified.

3. **Accuracy Score:**
   - The accuracy score, which is the same as the overall accuracy mentioned above, is 0.9613 (96.13%).

4. **Classification Report:**
   - The classification report provides additional insights into the precision, recall, and F1-score for each class (infected and uninfected), along with metrics like support.
   - Precision measures the proportion of true positive predictions out of all positive predictions made by the model.
   - Recall measures the proportion of true positive predictions out of all actual positive instances in the dataset.
   - The F1-score is the harmonic mean of precision and recall, providing a balanced measure of a classifier's performance.
   - The macro and weighted average metrics provide overall performance measures across both classes.

5. **Confusion Matrix:**
   - The confusion matrix visualizes the model's performance by showing the count of correct and incorrect predictions for each class.
   - In this case, the confusion matrix indicates that out of 2780 infected images, 2639 were correctly classified as infected (true positives), while 141 were incorrectly classified as uninfected (false negatives).
   - Similarly, out of 2730 uninfected images, 2658 were correctly classified as uninfected (true negatives), while 72 were incorrectly classified as infected (false positives).

Overall, these results demonstrate the effectiveness of our model in accurately classifying malaria cell images, with high precision and recall rates for both infected and uninfected classes. The confusion matrix provides additional insights into the specific types of classification errors made by the model.

![fifth_image](https://github.com/Ashishlathkar77/Malaria-Cell-Image-Classification/blob/main/Visualizations/download%20(28).png)

---
