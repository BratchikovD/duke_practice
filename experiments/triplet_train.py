import json
from multiprocessing import freeze_support
from pathlib import Path
import os
import torch
from torchvision.datasets import ImageFolder
from torch import optim
from torchvision import transforms
from torchvision import models

from helpers.utils import log_to_file, get_all_embeddings, plot_embeddings
from pytorch_metric_learning import miners, distances, losses, samplers
from tqdm import tqdm
import torchreid



if __name__ == '__main__':
    freeze_support()

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    DEVICE = torch.device("cuda:0")
    TRIPLETS_TYPE = "hard"
    BATCH_SIZE = 32
    EPOCHS = 120
    SAVE_PATH = f'../results/TripletLoss_{EPOCHS}_{BATCH_SIZE}_{TRIPLETS_TYPE}'

    h, w = 256, 128

    train_transforms = [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    test_transforms = [
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'test': transforms.Compose(test_transforms)
    }

    train_set = ImageFolder(TRAIN_DIR, data_transforms['train'])
    test_set = ImageFolder(TEST_DIR, data_transforms['test'])
    identity_sampler = samplers.MPerClassSampler(train_set.classes, 4, batch_size=BATCH_SIZE)
    data_loader = {
        'train': torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=identity_sampler, num_workers=2,
                                             pin_memory=True, prefetch_factor=2, persistent_workers=True),
        'test': torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=2,
                                            pin_memory=True, prefetch_factor=2, persistent_workers=True)
    }
    model = models.resnet50(pretrained=True).cuda()
    accuracy = torchreid.metrics.accuracy
    distance = distances.CosineSimilarity()
    criterion = losses.TripletMarginLoss(margin=0.3)
    mining_func = miners.TripletMarginMiner(margin=0.3, type_of_triplets=TRIPLETS_TYPE)
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.00035},
    ], weight_decay=0.0005)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    history = {"train": [], "val": [], "best_accuracy": 0.0}

    os.makedirs(SAVE_PATH, exist_ok=True)
    if os.path.exists(f"{SAVE_PATH}/training.log"):
        os.remove("training.log")

    for epoch in range(EPOCHS):
        model.to(DEVICE)
        model.train()

        for index, (inputs, labels) in (enumerate(tqdm(data_loader['train'], desc="Training", leave=False))):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            embeddings = model(inputs)

            triplets = mining_func(embeddings, labels)

            loss = criterion(embeddings, labels, triplets)
            loss.backward()

            optimizer.step()

            if index % 50 == 0 or index == len(data_loader['train']):
                history["train"].append({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "triplets": mining_func.num_triplets
                })
                msg = f"Эпоха [{epoch}/{EPOCHS}] Итерация [{index}/{len(data_loader['train'])}, Loss: {loss.item()}, Triplets: {mining_func.num_triplets}\]\n"
                log_to_file(msg)

        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()

            with torch.no_grad():
                embeddings, labels = get_all_embeddings(train_set, model, DEVICE)
                train_accuracy = accuracy(embeddings, labels, topk=(1, ))
                print(train_accuracy)
                plot_embeddings(embeddings, labels, epoch, SAVE_PATH)
                history["val"].append({"epoch": epoch, "accuracy": train_accuracy})
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
