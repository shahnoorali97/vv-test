import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.nn.functional import softmax

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
                all_preds = []
                all_labels = []
                all_probs = []

                for data in test_dataloader:
                    inputs, labels = data
                    outputs = model(inputs)

                    # Collect predicted probabilities for AUC-ROC calculation
                    probs = softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())

                     # Get predictions and append to list
                    preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

                    test_accuracy.append(
                        len(np.where(np.argmax(outputs.detach(), axis=1) == labels)[0])
                        / len(labels)
                    )
                    test_loss.append(loss(outputs, labels))

                    # Flatten the lists for metric calculations
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                all_probs = np.concatenate(all_probs)

                # Calculate metrics
                precision = precision_score(all_labels, all_preds, average='macro')
                recall = recall_score(all_labels, all_preds, average='macro')
                f1 = f1_score(all_labels, all_preds, average='macro')

                 # AUC-ROC calculation
                try:
                    auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                except ValueError:
                    auc_roc = None  # Handle cases where AUC cannot be computed

                message = (
                    f"\nEpoch: {epoch}\nMean training loss: {np.mean(training_loss)}"
                    f"\nLearning rate: {[item['lr'] for item in optimizer.param_groups][0]}"
                    f"\nMean validation loss: {np.mean(test_loss)}"
                    f"\nMean validation accuracy: {np.mean(test_accuracy)}"
                    f"\nPrecision: {precision:.4f}"
                    f"\nRecall: {recall:.4f}"
                    f"\nF1 Score: {f1:.4f}"
                )

                if auc_roc is not None:
                    message += f"\nAUC-ROC: {auc_roc:.4f}"
                else:
                    message += "\nAUC-ROC: Could not be computed"

                print(message)
