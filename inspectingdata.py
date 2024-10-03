import torch
import torchvision
from collections import Counter
import matplotlib.pyplot as plt
from utils.data import create_task3_datasets # create and return two dataloaders

# Create dataloaders
training_dataloader, test_dataloader = create_task3_datasets()

def inspect_dataset(dataloader):
    all_labels = []
    for images, labels in dataloader:
        all_labels.extend(labels.tolist())  # Collect all labels from the dataset

    # Get number of classes
    num_classes = len(set(all_labels))
    print(f"Number of classes: {num_classes}")

    # Get class distribution
    class_distribution = Counter(all_labels)
    print(f"Class distribution: {class_distribution}")

    # Plot class distribution for visualization
    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.title('Class Distribution')
    plt.show()

# Call the function to inspect your training dataset
inspect_dataset(training_dataloader)

# 2. Inspect image properties (image size, channels, etc.)
def inspect_image_properties(dataloader):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)  # Get a batch of images

    # Image batch shape: [batch_size, channels, height, width]
    print(f"Image batch shape: {images.shape}")

    # Get mean and std for normalization
    mean = images.mean([0, 2, 3])
    std = images.std([0, 2, 3])
    print(f"Mean of images (per channel): {mean}")
    print(f"Standard deviation of images (per channel): {std}")

    # Show some sample images
    grid = torchvision.utils.make_grid(images, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('Sample Training Images')
    plt.axis('off')
    plt.show()

# Call the function to inspect image properties
inspect_image_properties(training_dataloader)