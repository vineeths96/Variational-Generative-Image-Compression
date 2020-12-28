import torch
import argparse
from download_data import downloadData
from VAE_GAN import train
from validate_models import validate_models


def main():
    downloadData()

    arg_parser = argparse.ArgumentParser(description="Train model or test model (default)")
    arg_parser.add_argument("--train-model", action="store_true", default=False)

    argObj = arg_parser.parse_args()

    if argObj.train_model:
        encoder, generator = train()
        torch.save(encoder.state_dict(), f"../models/encoder_{NUM_CHANNELS}.model")
        torch.save(generator.state_dict(), f"../models/generator_{NUM_CHANNELS}.model")
    else:
        validate_models([2, 4, 8, 16, 28])


if __name__ == '__main__':
    main()
