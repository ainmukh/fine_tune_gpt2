import argparse
from torch.optim import Adam
from src.config_parser import ConfigParser
from src.utils import create_dataloaders, create_models, get_grouped_params
from src.trainer import Trainer


def main(config):
    dataloaders = create_dataloaders(config)
    model_16, model_32, tokenizer = create_models(config)

    optimizer = Adam(get_grouped_params(model_32, config), config['optimizer']['args'])

    trainer_class = Trainer(
        model_16, model_32,
        tokenizer, optimizer,
        config,
        dataloaders['train'], val_data_loader=dataloaders['valid']
    )
    trainer_class.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    config = ConfigParser.from_args(args)
    main(config)
