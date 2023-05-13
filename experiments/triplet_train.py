from multiprocessing import freeze_support
from pathlib import Path
from shutil import copyfile
import os
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import datasets, transforms
from torchvision import models
import time

from utils.dataloaders import TripletImageFolder
from utils.graphics import draw_curve
from utils.network_save import save_network

if __name__ == '__main__':
    freeze_support()

    version = torch.__version__

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    VAL_DIR = os.path.join(DATA_DIR, 'val')

    torch.cuda.set_device(0)
    use_gpu = torch.cuda.is_available()
    cudnn.benchmark = True

    h, w = 224, 224

    transform_train_list = [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    image_datasets = {'train': TripletImageFolder(TRAIN_DIR,
                                                  data_transforms['train']),
                      'val': TripletImageFolder(VAL_DIR,
                                                data_transforms['val'])}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                  shuffle=True, num_workers=2, pin_memory=True,
                                                  prefetch_factor=2, persistent_workers=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    y_loss = {'train': [], 'val': []}  # loss history
    y_err = {'train': [], 'val': []}


    def train_model(base_model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        warm_up = 0.1
        warm_iteration = round(dataset_sizes['train'] / 32) * 5

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    base_model.train(True)
                else:
                    base_model.train(False)

                running_loss = 0.0
                running_corrects = 0.0

                for iter, data in enumerate(dataloaders[phase]):
                    anchor_images, pos_images, neg_images, labels = data

                    now_batch_size, c, h, w = anchor_images.shape
                    if now_batch_size < 32:
                        continue

                    if use_gpu:
                        anchor_images = Variable(anchor_images.cuda().detach())
                        pos_images = Variable(pos_images.cuda().detach())
                        neg_images = Variable(neg_images.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                    else:
                        anchor_images, pos_images, neg_images, labels = Variable(anchor_images), Variable(
                            pos_images), Variable(neg_images), Variable(labels)

                    optimizer.zero_grad()

                    if phase == 'val':
                        with torch.no_grad():
                            anchor_outputs = model(anchor_images)
                            pos_outputs = model(pos_images)
                            neg_outputs = model(neg_images)
                    else:
                        anchor_outputs = model(anchor_images)
                        pos_outputs = model(pos_images)
                        neg_outputs = model(neg_images)

                    _, preds = torch.max(anchor_outputs.data, 1)
                    loss = criterion(anchor_outputs, pos_outputs, neg_outputs)

                    del anchor_images
                    del pos_images
                    del neg_images

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                    if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += loss.item() * now_batch_size
                    else:  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size

                    del loss
                    running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0 - epoch_acc)
                # deep copy the results
                if phase == 'val':
                    last_model_wts = base_model.state_dict()
                    if epoch % 10 == 9:
                        save_network(base_model, epoch)
                    draw_curve(epoch, y_loss, y_err, 'TripletLoss')
                if phase == 'train':
                    scheduler.step()
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))

        # load best results weights
        base_model.load_state_dict(last_model_wts)
        save_network(base_model, 'last')
        return base_model


    model = models.resnet50(pretrained=True).cuda()
    optim_name = optim.SGD

    optimizer_ft = optim_name([
        {'params': model.parameters(), 'lr': 0.1 * 0.05},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=25 * 2 // 3, gamma=0.1)
    dir_name = os.path.join(BASE_DIR, 'results', 'TripletLoss')

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copyfile('triplet_train.py', os.path.join(dir_name, 'train.py'))

    criterion = nn.TripletMarginLoss

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler)
