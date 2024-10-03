# Task 1:
# -------
# Analyze and engineer the training dataset (data/task1/train) to improve the model performance on the evaluation set.
# The training dataset and the training dataloader (defined in utils.data import create_task1_datasets) can be
# modified at will.
#
# However,
# * The training dataset should NOT be extended by external data
# * The test set should NOT be modified in any way
# * The model, loss, training loop, optimizer or any of the hyperparameters should NOT be modified
#
# With an improved training set, the model can be expected to reach validation accuracy of 0.88 and above.


import torch
import torchvision
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from utils.data import create_task1_datasets # create and return two dataloaders

# Create dataloaders
training_dataloader, test_dataloader = create_task1_datasets()

# Initialize the model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_classes = 10  # Update based on your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

# Inspect the shape of a batch to ensure it matches the model's expectations
data_iter = iter(training_dataloader)
images, labels = next(data_iter)
print(f'Image batch shape: {images.shape}')  # Should be [batch_size, 3, 224, 224]

# Training function
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=5):
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = train_correct / total_train
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')

# Train the model
train_model(model, training_dataloader, test_dataloader, optimizer, loss_fn, epochs=20)

# Visualize some training images
data_iter = iter(training_dataloader)
images, labels = next(data_iter)
grid = torchvision.utils.make_grid(images, nrow=8)
plt.imshow(grid.permute(1, 2, 0))
plt.title('Sample Training Images')
plt.axis('off')
plt.show()


#def create_task1_datasets():
    #train_dir = 'data/task1/train'
    #test_dir = 'data/task1/test'
    
    #train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    #est_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    #return train_loader, test_loader