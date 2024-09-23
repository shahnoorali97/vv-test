import torch
import numpy as np
from tqdm import tqdm


def train_model(
    model,
    training_dataloader,
    test_dataloader,
    optimizer,
    loss,
    epochs=50,
    test_interval=1,
):
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

                for data in test_dataloader:
                    inputs, labels = data
                    outputs = model(inputs)

                    test_accuracy.append(
                        len(np.where(np.argmax(outputs.detach(), axis=1) == labels)[0])
                        / len(labels)
                    )
                    test_loss.append(loss(outputs, labels))
                message = (
                    f"\nEpoch: {epoch}\nMean training loss: {np.mean(training_loss)}"
                    f"\nLearning rate: {[item['lr'] for item in optimizer.param_groups][0]}"
                    f"\nMean validation loss: {np.mean(test_loss)}"
                    f"\nMean validation accuracy: {np.mean(test_accuracy)}"
                )

                print(message)
