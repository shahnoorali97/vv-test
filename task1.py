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

from utils.models import ModelTask1
from utils.train import train_model
from utils.data import create_task1_datasets


training_dataloader, test_dataloader = create_task1_datasets()

model = ModelTask1()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
loss = torch.nn.CrossEntropyLoss()

train_model(model, training_dataloader, test_dataloader, optimizer, loss, epochs=50)
