import torch
import pandas as pd
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader

class CFG:
    batch_size = 16
    '''
    Help Functions
    '''

    @staticmethod
    def seed_everything(seed: int):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def loader(dataBase: pd.DataFrame, batch_size: int, n_workers=0,
               shuffle=False):
        return DataLoader(dataBase, batch_size=batch_size,
                          num_workers=n_workers, shuffle=shuffle)