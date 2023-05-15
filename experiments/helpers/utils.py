import os
import re
import torch
from matplotlib import pyplot as plt
from pytorch_metric_learning import testers
import numpy as np
from sklearn.manifold import TSNE


def get_all_embeddings(dataset, model, device):
    tester = testers.BaseTester(data_device=device)
    return tester.get_all_embeddings(dataset, model)


def log_to_file(msg, path="training.log"):
    with open(path, "a") as f:
        f.write(msg+"\n")


def get_accuracy(val_dataset, train_dataset, model, device):

    with torch.no_grad():
        val_embeddings, val_labels = get_all_embeddings(val_dataset, model, device=device)
        train_embeddings, train_labels = get_all_embeddings(train_dataset, model, device=device)
        # for each val embedding, find distance with all embeddings in train embeddings
        dist = torch.cdist(val_embeddings, train_embeddings)

    query_labels = val_labels.cpu().numpy()
    # Find index of closesest matching embedding
    matched_idx = torch.argmin(dist, axis=1).cpu().numpy()
    matched_labels = train_labels.cpu().numpy()[matched_idx]

    accuracy = (query_labels == matched_labels).mean()
    return accuracy


def plot_embeddings(embeddings, epoch, save_path, num_classes=20):
    tsne = TSNE(n_components=2, random_state=42)

    embeddings, labels = embeddings
    unique_classes = np.unique(labels)
    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)
    indices = np.isin(labels, selected_classes)
    embeddings = embeddings[indices]
    labels = labels[indices]

    embeddings_tsne = tsne.fit_transform(embeddings.cpu())

    cmap = plt.cm.get_cmap('rainbow', num_classes)

    plt.figure(figsize=(10,10))
    for idx, label in enumerate(labels):

        class_idx = np.where(selected_classes == label)[0][0]
        plt.scatter(*embeddings_tsne[idx], color=cmap(class_idx), label=str(label))
    plt.title(f'Embeddings at epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'embeddings_{epoch}.svg'))
