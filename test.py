import os
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from experiments.helpers.utils import get_all_embeddings

if __name__ == '__main__':
    freeze_support()
    IMAGE_SIZE = 224


    BASE_DIR = Path(__file__).parent
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    MODEL_PATH = "./results/TripletLoss_25_64_hard/model_best.pth"

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(MEAN, STD),
    ])

    val_dataset = ImageFolder(VAL_DIR, transform)
    test_dataset = ImageFolder(TEST_DIR, transform)

    DEVICE = torch.device("cuda:0")
    model = models.resnet50(pretrained=True).cuda()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # For all the images in test set, find most similar image in val set
    with torch.no_grad():
        # This is a faster way
        val_embeddings, val_labels = get_all_embeddings(val_dataset, model, device=DEVICE)
        test_embeddings, test_labels = get_all_embeddings(test_dataset, model, device=DEVICE)
        dist = torch.cdist(test_embeddings, val_embeddings)
    plt.figure(figsize=(20, 11))
    for i, idx in enumerate(np.random.choice(len(test_dataset), size=25, replace=False)):
        matched_idx = dist[idx].argmin().item()

        actual_label = test_labels[idx].item()
        predicted_label = val_labels[idx].item()

        actual_image_path = test_dataset.images[idx][0]
        predicted_image_path = val_dataset.images[matched_idx][0]

        actual_image = np.array(Image.open(actual_image_path).resize((IMAGE_SIZE, IMAGE_SIZE)))
        predicted_image = np.array(Image.open(predicted_image_path).resize((IMAGE_SIZE, IMAGE_SIZE)))
        stack = np.hstack([actual_image, predicted_image])

        plt.subplot(5, 5, i + 1)
        plt.imshow(stack)
        plt.title(f"GT: {actual_label}\nP: {predicted_label}", fontdict={'fontsize': 8})
        plt.axis("off")
