from .base import AbstractDataloader
from meantime.datasets import dataset_factory
from meantime.utils import all_subclasses
from meantime.utils import import_all_subclasses
import pdb
import_all_subclasses(__file__, __name__, AbstractDataloader)

DATALOADERS = {c.code():c
               for c in all_subclasses(AbstractDataloader)
               if c.code() is not None}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders() #喂入dataset，通过pytorch构建dataloader;
    return train, val, test


def get_dataloader(args):
    # pdb.set_trace()
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    return dataloader
