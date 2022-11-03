from datasets import load_dataset
from torch.utils.data import DataLoader


def create_dataloaders(config) -> dict:
    dataloaders = {}
    for split, params in config['data'].items():
        # load dataset
        dataset_params = params['dataset']
        path, name, key = dataset_params['path'], dataset_params['name'], dataset_params['key']
        dataset = load_dataset(path=path, name=name, split=split)
        dataset = dataset[key]

        batch_size = params['batch_size']

        # create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders[split] = dataloader
    return dataloaders
