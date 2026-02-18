import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns 

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def compute_confusion_matrix(preds, labels):
    return confusion_matrix (labels, preds)

def compute_per_class_accuracy(conf_matrix):
    return conf_matrix.diagonal() / conf_matrix.sum(axis = 1)

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize = (8, 6))
    sns.heatmap(confusion_matrix, annot = False, cmap = "Blues", xticklabels = class_names, yticklabels = class_names)

    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("confusion matrix")
    plt.show()


