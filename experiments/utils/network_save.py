import torch
import os


def save_network(network, epoch_label, experiment):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('../../results', experiment, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(0)
