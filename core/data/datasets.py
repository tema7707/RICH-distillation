from torch.utils.data import Dataset
from . import rich_utils
import numpy as np
import logging

log = logging.getLogger(__name__)


class ParticleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, ind):
        return  self.data[ind, rich_utils.y_count:-1], \
                self.data[ind, -1], \
                self.data[ind, :rich_utils.y_count]
