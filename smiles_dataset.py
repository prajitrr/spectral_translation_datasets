import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from smiles_embeddings import SmilesEmbeddings

import warnings
warnings.filterwarnings("ignore")

class SmilesDataset(Dataset):
    def __init__(self, csv_file, max_molecule_size, transform=None):
        self.smiles_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.embedder = SmilesEmbeddings(max_molecule_size=max_molecule_size)

    def __len__(self):
        return len(self.smiles_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        smiles = self.smiles_frame.iloc[idx, 0]
        try:
            image = self.embedder.embed_smiles(smiles).unsqueeze(0)
        except:
            image = self.embedder.embed_smiles("CCCCCCCC").unsqueeze(0)
        
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

