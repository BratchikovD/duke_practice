import argparse
import os

import torchreid

import models
from engines.arcface import ImageArcFaceEngine
from models.resnet import ResnetNewLosses
parser = argparse.ArgumentParser(
    prog='run_experiment',
    description='Запускает обучение и тестирование модели на датасете DukeMTMC.'
)

parser.add_argument('--model', default='resnet50',
                    help='Задаёт базовую модель для извлечения признаков. \nДоступные значения: resnet50.')
parser.add_argument('--loss', default='softmax',
                    help='Задает функцию потерь, которую нужно использовать при обучении. \nДоступные значения: softmax, triplet, arcface.')
parser.add_argument('--arc_scale', type=float, default=30.0, help='Feature_scale для ArcFace.')
parser.add_argument('--arc_margin', type=float, default=0.5, help='Смещение для ArcFace')
parser.add_argument('--optimizer', default='adam',
                    help='Выбор оптимизатора для обучения. \nДоступные значения: adam, sgd')
parser.add_argument('--scheduler', default='single_step',
                    help='Выбор lr_scheduler\'a, который использовать. \nДоступные значения: single_step, multi_step')
parser.add_argument('--sc_step_size', type=int, default=20, help='Scheduler шаг.')
parser.add_argument('--batch_size', type=int, default=32, help='Размера батча.')
parser.add_argument('--lr', type=float, default=0.0003, help='Задаёт learning rate для оптимизатора.')
parser.add_argument('--epochs', type=int, default=60, help='Количество эпох.')
parser.add_argument('--visualize', action='store_true', default=False, help='Визуализирует изображения результатов ранкинга.')
parser.add_argument('--log_path', help='Путь сохранения лога обучения.')

args = parser.parse_args()

datamanager = torchreid.data.ImageDataManager(
    root='.',
    sources='dukemtmcreid',
    height=256,
    width=128,
    batch_size_train=args.batch_size,
    batch_size_test=args.batch_size,
    combineall=False,
    train_sampler='RandomIdentitySampler' if args.loss == 'triplet' else 'RandomSampler'
)

if args.loss == 'arcface':
    model = models.build_model(
        name='resnet_arcface',
        num_classes=datamanager.num_train_pids,
        loss=args.loss
    )
else:
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss=args.loss
    )
model.cuda()
optimizer = torchreid.optim.build_optimizer(
    model,
    optim=args.optimizer,
    lr=args.lr
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler=args.scheduler,
    stepsize=args.sc_step_size
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
    engine = ImageArcFaceEngine(datamanager, model, optimizer, scheduler=scheduler, margin=args.arc_margin)
else:
    raise NotImplementedError

if not os.path.isdir('./logs'):
    os.mkdir('logs')

engine.run(
    max_epoch=args.epochs,
    save_dir=f'logs/resnet50-{args.loss}-{args.epochs}' if not args.log_path else args.log_path,
    print_freq=15
)

if args.visualize:
    engine.run(
        test_only=True,
        visrank=True,
    )
