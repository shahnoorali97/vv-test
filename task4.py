# Task 4:
# -------
# Implement a loss function that learns the classification task, but also minimizes the size of the output logits
# of the model. The (training) loss function and the associated parts of the training loop (e.g., loss function
# inputs) can be modified at will.
#
# However,
# * The model should NOT be modified
# * The evaluation loss function should NOT be modified


import numpy as np
import torch
from tqdm import tqdm

from utils.models import ModelTask4
from utils.data import create_task1_datasets


# Custom loss function with logit size penalty
def loss(outputs, labels, alpha=0.01):
    # Cross-entropy loss (for classification)
    ce_loss = torch.nn.functional.cross_entropy(outputs, labels)

    # Regularization based on logit size (L2 norm of logits)
    logit_penalty = torch.norm(outputs, p=2)

    # Combined loss with weighting factor alpha
    total_loss = ce_loss + alpha * logit_penalty

    return total_loss

training_dataloader, test_dataloader = create_task1_datasets()

model = ModelTask4()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
evaluation_loss = torch.nn.CrossEntropyLoss()

epochs = 100
test_interval = 1

for epoch in tqdm(range(epochs)):
    training_loss = []
    model.train()

    for data in training_dataloader:
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        # Use custom loss function
        l = loss(outputs, labels, alpha=0.01)
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
                logit_sizes.append(
                    np.mean(np.sqrt(np.power(outputs.detach().numpy(), 2)))
                )

                test_accuracy.append(
                    len(np.where(np.argmax(outputs.detach(), axis=1) == labels)[0])
                    / len(labels)
                )

                test_loss.append(evaluation_loss(outputs, labels).item())

            message = (
                f"\nEpoch: {epoch}\nMean training loss: {np.mean(training_loss)}"
                f"\nLearning rate: {[item['lr'] for item in optimizer.param_groups][0]}"
                f"\nMean validation loss: {np.mean(test_loss)}"
                f"\nMean validation accuracy: {np.mean(test_accuracy)}"
                f"\nMean logit size: {np.mean(logit_sizes)}"
            )

            print(message)
