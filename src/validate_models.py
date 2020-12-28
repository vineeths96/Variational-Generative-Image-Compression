import torch
from models import Encoder, Generator
from data_loader import test_dataloader
from utils import plot_image_grid
from parameters import *


def validate_models(channels):
    """
    Validate trained models
    :param channels: List of compressed channels used
    :return: None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = test_dataloader()
    test_batch = next(iter(test_loader)).to(device)

    reconstructed_images = {}

    for channel in channels:
        NUM_CHANNELS = channel
        encoder = Encoder(NUM_CHANNELS).to(device)
        generator = Generator(NUM_CHANNELS).to(device)

        encoder.load_state_dict(
            torch.load(f"../models/encoder_{NUM_CHANNELS}.model", map_location=torch.device("cpu"))
        )
        generator.load_state_dict(
            torch.load(f"../models/generator_{NUM_CHANNELS}.model", map_location=torch.device("cpu"))
        )

        encoder.eval()
        generator.eval()

        reconstructed_image = generator(encoder(test_batch))
        reconstructed_images[NUM_CHANNELS] = reconstructed_image

    plot_image_grid(test_batch, reconstructed_images, NUM_IMAGES_GRID)
