import torch.nn as nn
import matplotlib.pyplot as plt
from parameters import *


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)


def plot_image(test_batch, reconstructed_image, num_images):
    f, ax = plt.subplots(num_images, 2)

    for i in range(num_images):
        test_image = (test_batch[i].cpu().detach().permute(1, 2, 0) * STD) + MEAN
        rec_image = (reconstructed_image[i].cpu().detach().permute(1, 2, 0) * STD) + MEAN
        ax[i, 0].imshow(test_image)
        ax[i, 1].imshow(rec_image)
        f.set_figheight(20)
        f.set_figwidth(20)

    plt.show()
