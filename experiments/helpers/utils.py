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


def plot_embeddings(embeddings, labels, epoch, save_path):
    tsne = TSNE(n_components=2)
    embeddings_tsne = tsne.fit_transform(embeddings.cpu().numpy())

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(labels):
        ax.text(embeddings_tsne[i, 0], embeddings_tsne[i, 1], label, fontsize=8)
    plt.title(f"Embeddings at epoch {epoch}")
    plt.savefig(f"{save_path}/embeddings_epoch_{epoch}.png")
    plt.close()

    plt.savefig(os.path.join(save_path, f'embeddings_{epoch}.svg'))
