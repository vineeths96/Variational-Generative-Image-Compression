import torch
import argparse
from download_data import downloadData
from VAE_GAN import train
from validate_models import validate_models
from parameters import *


def main():
    """
    Main loop to train or validate model
    :return: None
    """

    downloadData()

    parser = argparse.ArgumentParser(description="Train model or test model (default)")
    parser.add_argument("--train-model", action="store_true", default=False)
    parser.add_argument("--num-channels", type=int, help="Number of channels in lowest dimension", default=8)

    arg_parser = parser.parse_args()

    if arg_parser.train_model:
        encoder, generator = train(channels=arg_parser.num_channels)
        torch.save(encoder.state_dict(), f"../models/encoder_{arg_parser.num_channels}.model")
        torch.save(generator.state_dict(), f"../models/generator_{arg_parser.num_channels}.model")
    else:
        validate_models(VALIDATE_CHANNELS)


if __name__ == "__main__":
    main()
