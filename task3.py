# Task 3:
# -------
# Analyze and engineer the training dataset (data/task3/train) to improve the model performance on the evaluation set.
# The training dataset and the training dataloader (defined in utils.data import create_task3_datasets) can be
# modified at will.
#
# However,
# * The training dataset should NOT be extended by external data
# * The test set should NOT be modified in any way
# * The model, loss, training loop, optimizer or any of the hyperparameters should NOT be modified
#
# With an improved training set, the model can be expected to reach validation accuracy of 0.84 and above.

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from utils.models import ModelTask3
from utils.train import train_model
from utils.data import create_task3_datasets
from torch.nn.functional import softmax

# Data Augmentation using transforms
augmentations = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),  # Converts the image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

# Load datasets with the augmentation
train_dataset = datasets.ImageFolder(root='data/task3/train', transform=augmentations)
test_dataset = datasets.ImageFolder(root='data/task3/train', transform=augmentations)

# Create dataloaders
training_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# If class_weights is returned, we can also modify the loss function to account for class imbalance.
loss = torch.nn.CrossEntropyLoss() # Weighted loss to handle imbalance

# Define the model and optimizer as given
model = ModelTask3()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)


train_model(model, training_dataloader, test_dataloader, optimizer, loss, epochs=20)

# ------------------------------ training ---------------------------------

def validate(model, validation_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():  # Disable gradient computation for validation
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Get predicted probabilities
            probs = softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            # Get the predicted class
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Flatten the lists
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # AUC-ROC for multi-class
    try:
        auc_roc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except ValueError:
        auc_roc = None  # Handle cases where AUC cannot be computed

    return val_loss / len(validation_loader), precision, recall, f1, auc_roc

