import json
from multiprocessing import freeze_support
from pathlib import Path
import os
import torch
from torchvision.datasets import ImageFolder
from torch import optim
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet50_Weights

from helpers.utils import log_to_file, get_accuracy, get_all_embeddings, plot_embeddings
from pytorch_metric_learning import miners, distances, losses
from tqdm import tqdm
import torchreid
if __name__ == '__main__':
    freeze_support()

    DEVICE = torch.device("cuda:0")
    TRIPLETS_TYPE = "hard"
    BATCH_SIZE = 64
    EPOCHS = 120
    SAVE_PATH = f'../results/TripletLoss_{EPOCHS}_{BATCH_SIZE}_{TRIPLETS_TYPE}'

    identity_sampler = torchreid.data.sampler.RandomIdentitySampler
    dataset = torchreid.data.datasets.image.dukemtmcreid.DukeMTMCreID()
    data_manager = torchreid.data.datamanager.ImageDataManager(root=".", sources='dukemtmcreid',
                                                               train_sampler='RandomIdentitySampler')

    model = models.resnet50(pretrained=True).cuda()
    accuracy = torchreid.metrics.accuracy.accuracy
    distance = distances.CosineSimilarity()
    criterion = losses.TripletMarginLoss(margin=0.3)
    mining_func = miners.TripletMarginMiner(margin=0.3, type_of_triplets=TRIPLETS_TYPE)
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.00035},
    ], weight_decay=0.0005)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    history = {"train": [], "best_accuracy": 0.0}

    os.makedirs(SAVE_PATH, exist_ok=True)
    if os.path.exists(f"{SAVE_PATH}/training.log"):
        os.remove("training.log")

    for epoch in range(EPOCHS):
        model.to(DEVICE)
        model.train()

        for index, (inputs, labels) in (enumerate(tqdm(data_manager.train_loader, desc="Training", leave=False))):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            embeddings = model(inputs)

            triplets = mining_func(embeddings, labels)

            loss = criterion(embeddings, labels, triplets)
            loss.backward()

            optimizer.step()

            if index % 50 == 0 or index == len(data_manager.train_loader):
                history["train"].append({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "triplets": mining_func.num_triplets
                })
                msg = f"Эпоха [{epoch}/{EPOCHS}] Итерация [{index}/{len(data_manager.train_loader)}, Loss: {loss.item()}, Triplets: {mining_func.num_triplets}\]\n"
                log_to_file(msg)

        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()

            with torch.no_grad():
                embeddings, labels = get_all_embeddings(dataset.train, model, DEVICE)

                plot_embeddings(embeddings, labels, epoch, SAVE_PATH)

                history["val"].append({"epoch": epoch, "accuracy": accuracy(embeddings, labels, topk=1)})
                msg = f"Train accuracy: {accuracy}"
                log_to_file(msg)
                print(msg)
                torch.save(model.state_dict(), f"{SAVE_PATH}/model_latest.pth")

                if accuracy >= history["best_accuracy"]:
                    history["best_accuracy"] = accuracy
                    torch.save(model.state_dict(), f"{SAVE_PATH}/model_best.pth")

                with open(f"{SAVE_PATH}/history.json", "w") as f:
                    f.write(json.dumps(history))
        scheduler.step()
