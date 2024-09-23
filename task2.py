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
from tqdm import tqdm

from utils.models import ModelTask1
from utils.data import create_task1_datasets


training_dataloader, test_dataloader = create_task1_datasets()

model = ModelTask1()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
loss = torch.nn.CrossEntropyLoss()

epochs = 50
test_interval = 1

for epoch in tqdm(range(epochs)):
    training_loss = []
    model.train()

    for data in training_dataloader:
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        l = loss(outputs, labels)
        l.backward()

        training_loss.append(l.detach())

        optimizer.step()

    if ((epoch + 1) % test_interval) == 0:
        model.eval()

        with torch.no_grad():
            test_loss = []
            test_accuracy = []
            logit_sizes = []

            for data in test_dataloader:
                inputs, labels = data
                outputs = model(inputs)
                logit_sizes.append(np.mean(np.sqrt(np.power(outputs.detach().numpy(), 2))))

                test_accuracy.append(
                    len(np.where(np.argmax(outputs.detach(), axis=1) == labels)[0])
                    / len(labels)
                )

                test_loss.append(loss(outputs, labels))

            message = (f"\nEpoch: {epoch}\nMean training loss: {np.mean(training_loss)}"
                       f"\nLearning rate: {[item['lr'] for item in optimizer.param_groups][0]}"
                       f"\nMean validation loss: {np.mean(test_loss)}"
                       f"\nMean validation accuracy: {np.mean(test_accuracy)}")

            print(message)
