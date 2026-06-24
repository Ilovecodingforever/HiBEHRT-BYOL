from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from dataloaders import transform
from torchvision import transforms
import pandas as pd


def load_hibehrt_dataframe(path):
    """Load HiBEHRT adapter outputs from pickle when available, else parquet."""
    return pd.read_pickle(path) if str(path).endswith((".pkl", ".pickle")) else pd.read_parquet(path)


def _long_tensor(value):
    """Convert nested sequence outputs into one contiguous int64 tensor."""
    return torch.as_tensor(np.asarray(value), dtype=torch.long)


class SSLDset(Dataset):
    def __init__(self, dataset, params):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.data = dataset
        self._compose = transforms.Compose([
            transform.MordalitySelection(params['mordality']),
            transform.RandomKeepDiagMed(),
            transform.RandomCropSequence(p=params['p'], seq_threshold=params['seq_threshold']),
            transform.TruncateSeqence(params['max_seq_length']),
            transform.EHRAugmentation(),
            transform.CreateSegandPosition(),
            # transform.RemoveSEP(),
            transform.TokenAgeSegPosition2idx(params['token_dict_path'], params['age_dict_path']),
            transform.RetriveSeqLengthAndPadding(params['max_seq_length']),
            transform.FormatAttentionMask(params['max_seq_length']),
            transform.FormatHierarchicalStructure(params['segment_length'], params['move_length'],
                                                  params['max_seq_length']),
            transform.CalibrateHierarchicalPosition(),
            transform.CalibrateSegmentation()
        ])

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        sample = {'code': self.data.code[index],
                  'age': self.data.age[index]
                  # 'seg': self.data.seg[index],
                  # 'position': self.data.position[index]
                  }

        sample = self._compose(sample)

        return {'code': _long_tensor(sample['code']),
                'age': _long_tensor(sample['age']),
                'seg': _long_tensor(sample['seg']),
                'position': _long_tensor(sample['position']),
                'att_mask': _long_tensor(sample['att_mask']),
                'h_att_mask': _long_tensor(sample['h_att_mask'])}

    def __len__(self):
        return len(self.data)


def MlmByolDataLoader(params):
    if params['data_path'] is not None:
        data = load_hibehrt_dataframe(params['data_path'])
        if 'fraction' in params:
            data = data.sample(frac=params['fraction'], random_state=0).reset_index(drop=True)
        print('number of patients:', len(data))
        dset = SSLDset(dataset=data, params=params)
        dataloader = DataLoader(dataset=dset,
                                batch_size=params['batch_size'],
                                shuffle=params['shuffle'],
                                num_workers=params['num_workers']
                                )
        return dataloader
    else:
        return None