import os
import re
from collections import defaultdict
import random
import torch
from matplotlib import pyplot as plt
from pytorch_metric_learning import testers
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm as cm


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


def plot_embeddings(embeddings, labels, epoch, save_path, class_nums=20):

    embeddings_by_class = defaultdict(list)
    for embedding, label in zip(embeddings, labels):
        embeddings_by_class[label.item()].append(embedding.cpu().numpy())

    random_classes = random.sample(list(embeddings_by_class.keys()), class_nums)

    random_embeddings = {class_name: embeddings_by_class[class_name] for class_name in random_classes}

    all_random_embeddings = []
    all_labels = []
    for class_name, embeddings in random_embeddings.items():
        for embedding in embeddings:
            all_random_embeddings.append(embedding)
            all_labels.append(class_name)

    embeddings_2d = TSNE(n_components=2).fit_transform(all_random_embeddings)

    unique_labels = list(set(all_labels))
    colors = cm.tab20(np.arange(len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(embeddings_2d.shape[0]):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color_map[all_labels[i]])
    plt.title(f"Embeddings at epoch {epoch}")
    plt.savefig(f"{save_path}/embeddings_epoch_{epoch}.svg")
    plt.close()
