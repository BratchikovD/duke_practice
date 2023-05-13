import os
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent.parent

def draw_curve(current_epoch, y_loss, y_err, experiment):
    x_epoch = list(range(current_epoch + 1))

    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')

    if current_epoch == 0:
        ax0.legend()
        ax1.legend()

    plt.savefig(os.path.join(BASE_DIR, 'results', experiment, 'train.jpg'))
