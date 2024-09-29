# Task 2:
# -------
# Improve the training loop by implementing the following:
# * Periodic model checkpoints
# * Early stoppage on plateauing or increasing validation loss
# * Dynamic learning rate scheduler
# * Stochaistic Weight Averaging
#
# Any external libraries can be used, but the use of those in the torch-package is preferred.


import numpy as np
import torch
import os
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR

from utils.models import ModelTask1
from utils.data import create_task1_datasets

# Set paths for model saving
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

training_dataloader, test_dataloader = create_task1_datasets()

model = ModelTask1()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 50
test_interval = 1
early_stopping_patience = 5  # Stop after 5 epochs without improvement
best_val_loss = np.inf
patience_counter = 0

# SWA
swa_model = AveragedModel(model)
swa_start_epoch = 30  # When to start using SWA
swa_scheduler = SWALR(optimizer, swa_lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
                                     
for epoch in tqdm(range(epochs)):
    training_loss = []
    model.train()

    for inputs, labels in training_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())

    if ((epoch + 1) % test_interval) == 0:
        model.eval()
        test_loss = []
        test_accuracy = []

        with torch.no_grad():
            # logit_sizes = []
            for inputs, labels in test_dataloader:
                outputs = model(inputs)
                val_loss = loss_fn(outputs, labels)
                test_loss.append(val_loss.item())
                preds = torch.argmax(outputs, dim=1)
                accuracy = (preds == labels).float().mean().item()
                test_accuracy.append(accuracy)

        mean_train_loss = np.mean(training_loss)
        mean_test_loss = np.mean(test_loss)
        mean_test_accuracy = np.mean(test_accuracy)
        
        # ReduceLROnPlateau based on validation loss
        if epoch < swa_start_epoch:
            scheduler.step(mean_test_loss)
        else:
            cosine_scheduler.step()

        # Save model checkpoints periodically
        checkpoint_path = f'{checkpoint_dir}/model_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_path)

        print(f"\nEpoch: {epoch+1}\nMean training loss: {mean_train_loss:.4f}")
        print(f"Mean validation loss: {mean_test_loss:.4f}")
        print(f"Mean validation accuracy: {mean_test_accuracy:.4f}")
        print(f"Learning rate: {[param_group['lr'] for param_group in optimizer.param_groups][0]}")

        # Early stopping check
        if mean_test_loss < best_val_loss:
            best_val_loss = mean_test_loss
            patience_counter = 0
            print("Validation loss improved, resetting patience.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    # Stochastic Weight Averaging (SWA)
    if epoch >= swa_start_epoch:
        swa_model.update_parameters(model)
        swa_scheduler.step()

# Finalize SWA model and evaluate
torch.optim.swa_utils.update_bn(training_dataloader, swa_model)  # Update batch normalization for SWA
swa_checkpoint_path = f'{checkpoint_dir}/swa_model_final.pt'
torch.save(swa_model.state_dict(), swa_checkpoint_path)

print(f"SWA model saved at {swa_checkpoint_path}")

# Evaluate SWA model performance
def evaluate_swa_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

# Load the SWA model and evaluate
swa_model.load_state_dict(torch.load(swa_checkpoint_path))
evaluate_swa_model(swa_model, test_dataloader)