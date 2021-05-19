import random

import numpy as np
import torch

from torch_utils_rich_mrartemev import *

class RichDataset(torch.utils.data.Dataset):
    def __init__(
        self
        , data
    ):
        super(RichDataset).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    # Гененрирует два случайных семпла, забиваем на индекс
    def __getitem__(self, _):
        idx1 = random.randint(0, self.data.shape[0] - 1)
        idx2 = random.randint(0, self.data.shape[0] - 1)
        idx3 = random.randint(0, self.data.shape[0] - 1)
        return (self.data[idx1], self.data[idx2], self.data[idx3])

# Хотим разбить на куски: dll + вход + веса, и сгенерить noise
# ->: настоящий выход + вход 1, шум 1 + вход 1, шум 2 + вход 2, веса 1, веса 2
class collate_fn_rich:
    def __init__(self, device):
        self.device = device 

    # (arr1, arr2)
    def __call__(self, samples):
        batch_size = len(samples)

        full_1 = torch.cat([torch.tensor(t1).unsqueeze(0) for (t1, t2, t3) in samples], dim=0)
        full_2 = torch.cat([torch.tensor(t2).unsqueeze(0) for (t1, t2, t3) in samples], dim=0)
        full_3 = torch.cat([torch.tensor(t3).unsqueeze(0) for (t1, t2, t3) in samples], dim=0)

        input_1 = full_1[:, y_count:-1]
        input_2 = full_2[:, y_count:-1]
        real = full_3[:, :-1]

        w_1 = full_1[:, -1]
        w_2 = full_2[:, -1]
        w_real = full_3[:, -1]

        return (
            real.to(self.device)
            , input_1.to(self.device)
            , input_2.to(self.device)
            , w_real.to(self.device)
            , w_1.to(self.device)
            , w_2.to(self.device)
        )


class SingleDataset(torch.utils.data.Dataset):
    def __init__(
        self
        , data
    ):
        super(RichDataset).__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

class collate_fn_single:
    def __init__(self, device):
        self.device = device

    def __call__(self, samples):
        batch_size = len(samples)

        full = torch.cat([torch.tensor(t).unsqueeze(0) for t in samples], dim=0)

        input_features = full[:, y_count:-1]

        w = full[:, -1]
        real = full[:, :-1]

        return (
            real.to(self.device)
            , input_features.to(self.device)
            , w.to(self.device)
        )
