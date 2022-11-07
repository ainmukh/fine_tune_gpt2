import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import logging

LOG_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
NO_DECAY = ['bias', 'LayerNorm.weight']


def create_dataloaders(config) -> dict:
    dataloaders = {}
    for split, params in config['data'].items():
        # load dataset
        dataset_params = params['dataset']
        path, name, key = dataset_params['path'], dataset_params['name'], dataset_params['key']
        if path.startswith('codeparrot'):
            cur_path = path + '-'
            split_for_path = split if split == 'train' else 'valid'
            cur_path = cur_path + split_for_path
            dataset = load_dataset(path=cur_path, name=name, split='train')
        else:
            dataset = load_dataset(path=path, name=name, split=split)
        # dataset = dataset.unique(key)

        batch_size = dataset_params['batch_size']

        # create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders[split] = dataloader
    return dataloaders


def get_grouped_params(model, config):
    params_with_wd, params_without_wd = [], []
    for name, p in model.named_parameters():
        if any(p_no_decay in name for p_no_decay in NO_DECAY):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)

    weight_decay = config['optimizer']['args']['weight_decay']
    return [{'params': params_with_wd, 'weight_decay': weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]


def create_models(config):
    model_config = config['model']
    model_name = model_config['model_name']

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # gpu model
    model_16 = transformers.GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model_16.gradient_checkpointing_enable()
    model_16.cuda()

    # cpu model
    model_32 = transformers.GPT2LMHeadModel.from_pretrained(model_name)
    return model_16, model_32, tokenizer


def get_logger(name, verbosity: int = 2):
    msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
        verbosity, LOG_LEVELS.keys()
    )
    assert verbosity in LOG_LEVELS, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[verbosity])
    # print(type(logger))  class 'logging.Logger'
    return logger


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
