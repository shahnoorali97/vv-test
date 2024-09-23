# vv-test

## General

The technical supplementary tasks consists of four problems in the topics of data analysis, model training frameworks, data engineering and loss functions. Each task is a self-contained script that trains a machine learning model wrt. 
task specifics. The preferred way of completing the tasks is by implementing the changes directly to the scripts.

### Publishing the solutions

The preferred method of publishing the result is publishing under your own Github account and providing the URL to your contact person. Other methods are acceptable as well.

### Deadline

The tasks can be done at your convenience. Each task is intended to take less than 1h to complete, but no specific completion time is given or enforced.

### Partial solutions

It should be remembered that the tasks are a tool for evaluating technical and problem solving skills. We value a logical and structured approach at solving the tasks as much as a complete, finished solution.
If some task is not progressing, it may be preferable to submit a partial solution with a description of the attempt at solution, than spend excessive time on forcing a complete solution.

## Prerequisites
Clone the project: ```git clone https://github.com/llaakson-vv/vv-test.git```, extract the data folder: ```tar -xf data.tar.xz```, create a Python virtual environment, and install dependencies: ```pip install -r requirements```

## Task 1:

Analyze and engineer the training dataset (data/task1/train) to improve the model performance on the evaluation set.
The training dataset and the training dataloader (defined in utils.data import create_task1_datasets) can be
modified at will.

However,
* The training dataset should NOT be extended by external data
* The test set should NOT be modified in any way
* The model, loss, training loop, optimizer or any of the hyperparameters should NOT be modified

With an improved training set, the model can be expected to reach validation accuracy of 0.88 and above.

## Task 2:

Improve the training loop by implementing the following:
* Periodic model checkpoints
* Early stoppage on plateauing or increasing validation loss
* Dynamic learning rate scheduler
* Stochaistic Weight Averaging

Any external libraries can be used, but the use of those in the torch-package is preferred.

## Task 3:

Analyze and engineer the training dataset (data/task3/train) to improve the model performance on the evaluation set.
The training dataset and the training dataloader (defined in utils.data import create_task3_datasets) can be
modified at will.

However,
* The training dataset should NOT be extended by external data
* The test set should NOT be modified in any way
* The model, loss, training loop, optimizer or any of the hyperparameters should NOT be modified

With an improved training set, the model can be expected to reach validation accuracy of 0.84 and above.

## Task 4:

Implement a loss function that learns the classification task, but also minimizes the size of the output logits
of the model. The (training) loss function and the associated parts of the training loop (e.g., loss function
inputs) can be modified at will.

However,
* The model should NOT be modified
* The evaluation loss function should NOT be modified
