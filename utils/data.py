import os
import torch
import cv2

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def read_mnist_data(root):
    data = []

    for label in os.listdir(root):
        for sample in filter(
            lambda s: s.endswith(".png"), os.listdir(f"{root}/{label}")
        ):
            img = cv2.imread(f"{root}/{label}/{sample}", cv2.IMREAD_GRAYSCALE)
            data.append((torch.from_numpy(img).float().unsqueeze(dim=0), int(label)))

    return data


def read_synthetic_data(root):
    tensors = []

    labels = open(f"{root}/labels.csv", "r").read()
    data = {}
    for item in labels.split(","):
        data[item.split(":")[0]] = float(item.split(":")[1])

    for item in filter(lambda s: s.endswith(".png"), os.listdir(root)):
        img = cv2.imread(f"{root}/{item}", cv2.IMREAD_GRAYSCALE)
        theta = data[item.replace(".png", "")]

        tensors.append((torch.from_numpy(img).float().unsqueeze(dim=0), theta))

    return tensors


def create_task2_datasets():
    training_set = read_mnist_data("data/task1/train")
    validation_set = read_mnist_data("data/task1/test")

    training_loader = DataLoader(training_set, batch_size=256)
    validation_loader = DataLoader(validation_set, batch_size=256, shuffle=True)

    return training_loader, validation_loader

def create_task1_datasets():
    
    # Data augmentation for training
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    train_dir = 'data/task1/train'
    test_dir = 'data/task1/test'
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, test_loader

def create_task2_datasets():
    training_set = read_mnist_data("data/task1/train")
    validation_set = read_mnist_data("data/task1/test")

    training_loader = DataLoader(training_set, batch_size=256)
    validation_loader = DataLoader(validation_set, batch_size=256, shuffle=True)

    return training_loader, validation_loader

def create_task3_datasets():
    training_set = read_mnist_data("data/task3/train")
    validation_set = read_mnist_data("data/task3/test")

    training_loader = DataLoader(
        training_set,
        batch_size=256,
        shuffle=True,
    )
    validation_loader = DataLoader(validation_set, batch_size=256, shuffle=False)

    return training_loader, validation_loader
