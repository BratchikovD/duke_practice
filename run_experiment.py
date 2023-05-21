import argparse
import os

import torchreid

import models
from engines import ImageArcFaceEngine, ContrastiveEngine, CenterLossEngine, TripletCenterEngine, SphereFaceEngine

parser = argparse.ArgumentParser(
    prog='run_experiment',
    description='Запускает обучение и тестирование модели на датасете DukeMTMC.'
)

parser.add_argument('--model', default='resnet_arcface',
                    help='Задаёт базовую модель для извлечения признаков. \nДоступные значения:  resnet_arcface, resnet152.')
parser.add_argument('--loss', default='softmax',
                    help='Задает функцию потерь, которую нужно использовать при обучении. \nДоступные значения: softmax, triplet, arcface, contrastive, triplet_center.')
parser.add_argument('--optimizer', default='adam',
                    help='Выбор оптимизатора для обучения. \nДоступные значения: adam, sgd, amsgrad')
parser.add_argument('--scheduler', default='single_step',
                    help='Выбор lr_scheduler\'a, который использовать. \nДоступные значения: single_step, multi_step')
parser.add_argument('--sc_step_size', type=int, default=20, help='Scheduler шаг.')
parser.add_argument('--gamma', type=float, default=0.1, help='Множитель learning_rate для scheduler')
parser.add_argument('--batch_size', type=int, default=32, help='Размера батча.')
parser.add_argument('--lr', type=float, default=0.0003, help='Задаёт learning rate для оптимизатора.')
parser.add_argument('--epochs', type=int, default=60, help='Количество эпох.')
parser.add_argument('--visualize', action='store_true', default=False, help='Визуализирует изображения результатов ранкинга.')
parser.add_argument('--log_path', help='Путь сохранения лога обучения.')

args = parser.parse_args()

identity_sampler_losses = ['triplet', 'contrastive', 'triplet_center']
datamanager = torchreid.data.ImageDataManager(
    root='.',
    sources='dukemtmcreid',
    height=256,
    width=128,
    transforms=['random_flip', 'random_crop'],
    batch_size_train=args.batch_size,
    batch_size_test=256,
    combineall=False,
    train_sampler='RandomIdentitySampler' if args.loss in identity_sampler_losses else 'RandomSampler',
    num_instances=6,
)

model = models.build_model(
    name=args.model,
    num_classes=datamanager.num_train_pids,
    loss=args.loss,
    pretrained=True
)
model.cuda()
print(model)
optimizer = torchreid.optim.build_optimizer(
    model,
    optim=args.optimizer,
    lr=args.lr,
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[60, 80],
    max_epoch=args.epochs,
    gamma=args.gamma,
)

if args.loss == 'triplet':
    engine = torchreid.engine.ImageTripletEngine(
        datamanager, model, optimizer, margin=0.3,
        weight_t=1, weight_x=0, scheduler=scheduler
    )
elif args.loss == 'softmax':
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer, scheduler=scheduler
    )
elif args.loss == 'arcface':
    engine = ImageArcFaceEngine(datamanager, model, optimizer, scheduler=scheduler)
elif args.loss == 'contrastive':
    engine = ContrastiveEngine(datamanager, model, optimizer, scheduler=scheduler)
elif args.loss == 'center':
    engine = CenterLossEngine(datamanager, model, optimizer, scheduler=scheduler)
elif args.loss == 'triplet_center':
    engine = TripletCenterEngine(datamanager, model, optimizer, scheduler=scheduler, weight_center=0.01, weight_triplet=1)
elif args.loss == 'sphere':
    engine = SphereFaceEngine(datamanager, model, scheduler=scheduler)
else:
    raise NotImplementedError

if not os.path.isdir('./logs'):
    os.mkdir('logs')

engine.run(
    max_epoch=args.epochs,
    save_dir=f'logs/resnet50-{args.loss}-{args.epochs}' if not args.log_path else args.log_path,
    print_freq=15,
)

if args.visualize:
    engine.run(
        test_only=True,
        visrank=True,
    )
