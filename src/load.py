import os
import numpy as np
from torch.utils.data import Dataset
import torch


class FrameDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.subjects = [subject.replace('_frames.npy', '') for subject in os.listdir(data_dir) if
                         subject.endswith('_frames.npy')]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        frames = np.load(os.path.join(self.data_dir, f'{subject}_frames.npy'))
        gt = np.load(os.path.join(self.data_dir, f'{subject}_gt.npy'))

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return torch.stack(frames), torch.tensor(gt, dtype=torch.float32)
