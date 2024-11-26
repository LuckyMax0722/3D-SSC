import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

class SemanticKittiLabelDataset(Dataset):
    def __init__(self, 
                 split,
                 data_root,):
        """
        Args:
            split: train or val
            data_root: /path/to/semantic_kitti_dataset
        """
        
        self.label_root = os.path.join(data_root, 'labels')
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }

        self.split = split
        self.sequences = splits[split]
        
        self.file_paths = self._get_all_files()
        

    def _get_all_files(self):
        """
        get all `_1_2.npy` file path。
        """
        
        file_paths = []
        for sequence in self.sequences:
            sequence_path = os.path.join(self.label_root, sequence)
            
            if not os.path.exists(sequence_path):
                continue
            
            for file_name in os.listdir(sequence_path):
                if file_name.endswith("_1_2.npy"):
                    file_paths.append(os.path.join(sequence_path, file_name))
                    
        return sorted(file_paths)
    
    def _get_target_1_2(self, idx):
        if self.split == 'train' or self.split == 'val':
            file_path = self.file_paths[idx]
            target_1_2 = np.load(file_path)
            target_1_2 = target_1_2.reshape(128, 128, 16).astype(np.float32)
            
            target_1_2[target_1_2 == 255] = 20  # class 21
        else:
            target_1_2 = None

        data = dict(
            target_1_2 = target_1_2
        )
        
        return data
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = self._get_target_1_2(idx)
        
        return data


class SemanticKittiLabelDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = SemanticKittiLabelDataset(split='train', data_root=self.data_root)
        self.val_dataset = SemanticKittiLabelDataset(split='val', data_root=self.data_root)

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


if __name__ == "__main__":
    train_dataset = SemanticKittiLabelDataset(split='train', data_root=CONF.PATH.DATA_DATASETS)
    train_dataset[0]
    data_module = SemanticKittiLabelDataModule(data_root=CONF.PATH.DATA_DATASETS, batch_size=32, num_workers=4)


    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    '''
    for batch in train_loader:
        print(batch)
        break
    '''

