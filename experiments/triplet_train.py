import json
from multiprocessing import freeze_support
from pathlib import Path
import os
import torch
from torchvision.datasets import ImageFolder
from torch import optim
from torchvision import transforms
from torchvision import models
from helpers.utils import log_to_file, get_accuracy
from pytorch_metric_learning import miners, distances, losses
from tqdm import tqdm


if __name__ == '__main__':
    freeze_support()

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')

    h, w = 224, 224

    transform_train_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
    }

    train_dataset = ImageFolder(TRAIN_DIR, data_transforms['train'])
    val_dataset = ImageFolder(VAL_DIR, data_transforms['train'])

    BATCH_SIZE = 16
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=2, pin_memory=True,
                                             prefetch_factor=2, persistent_workers=True)

    dataset_size = len(train_dataset)
    class_names = train_dataset.classes

    DEVICE = torch.device("cuda:0")
    TRIPLETS_TYPE = "semihard"
    model = models.resnet50(pretrained=True).cuda()
    distance = distances.CosineSimilarity()
    criterion = losses.TripletMarginLoss(margin=0.2, distance=distance)
    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets=TRIPLETS_TYPE)
    optimizer = optim.SGD([
        {'params': model.parameters(), 'lr': 0.001},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    EPOCHS = 25
    SAVE_PATH = f'../results/TripletLoss_{EPOCHS}_{BATCH_SIZE}_{TRIPLETS_TYPE}'
    history = {"train": [], "val": [], "best_accuracy": 0.0}

    os.makedirs(SAVE_PATH, exist_ok=True)
    if os.path.exists(f"{SAVE_PATH}/training.log"):
        os.remove("training.log")

    for epoch in range(EPOCHS):
        model.to(DEVICE)
        model.train()

        for index, (inputs, labels) in (enumerate(tqdm(dataloader, desc="Training", leave=False))):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            embeddings = model(inputs)

            triplets = mining_func(embeddings, labels)

            loss = criterion(embeddings, labels, triplets)
            loss.backward()

            optimizer.step()

            if index % 100 == 0:
                history["train"].append({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "triplets": mining_func.num_triplets
                })
                msg = f"Эпоха [{epoch}/{EPOCHS}] Итерация [{index}/{len(dataloader)}, Loss: {loss.item()}, Triplets: {mining_func.num_triplets}\]\n"
                log_to_file(msg)

        if epoch % 2 == 0:
            model.eval()

            with torch.no_grad():
                accuracy = get_accuracy(val_dataset, train_dataset, model, DEVICE)

                history["val"].append({"epoch": epoch, "accuracy": accuracy})
                msg = f"Val accuracy: {accuracy}"
                log_to_file(msg)

                torch.save(model.state_dict(), f"{SAVE_PATH}/model_latest.pth")

                if accuracy >= history["best_accuracy"]:
                    history["best_accuracy"] = accuracy
                    torch.save(model.state_dict(), f"{SAVE_PATH}/model_best.pth")

                with open(f"{SAVE_PATH}/history.json", "w") as f:
                    f.write(json.dumps(history))
