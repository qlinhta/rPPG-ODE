import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm


class RPpgDataset(Dataset):
    def __init__(self, features_dir, gt_dir, max_channels):
        self.features_dir = features_dir
        self.gt_dir = gt_dir
        self.max_channels = max_channels
        self.subjects = [f.replace('_features.npy', '') for f in os.listdir(features_dir) if
                         f.endswith('_features.npy')]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        features = np.load(os.path.join(self.features_dir, f"{subject}_features.npy"))
        gt = np.load(os.path.join(self.gt_dir, f"{subject}_gt.npy"))

        if features.shape[-1] > self.max_channels:
            features = features[..., :self.max_channels]
        elif features.shape[-1] < self.max_channels:
            pad_size = self.max_channels - features.shape[-1]
            features = np.pad(features, ((0, 0), (0, 0), (0, 0), (0, pad_size)), mode='constant')
        features = features.reshape(features.shape[0], -1)
        gt_value = np.mean(gt)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(gt_value, dtype=torch.float32)


def collate_fn(batch):
    features, gt = zip(*batch)
    max_len_features = max([f.shape[0] for f in features])

    padded_features = []

    for f in features:
        pad_size_features = max_len_features - f.shape[0]

        padded_f = torch.nn.functional.pad(f, (0, 0, 0, pad_size_features))

        padded_features.append(padded_f)

    padded_features = torch.stack(padded_features)
    gt = torch.stack(gt)

    return padded_features, gt


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def train_model(dataloader, rnn_model, criterion, optimizer, device):
    rnn_model.train()
    losses = []
    progress = tqdm(dataloader, desc="Training", unit="batch")
    for features, gt in progress:
        features, gt = features.to(device), gt.to(device)
        optimizer.zero_grad()
        outputs = rnn_model(features)
        loss = criterion(outputs, gt.view(-1, 1))
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=loss.item())
        losses.append(loss.item())
    return np.mean(losses)


def validate_model(dataloader, rnn_model, criterion, device):
    rnn_model.eval()
    losses = []
    with torch.no_grad():
        for features, gt in tqdm(dataloader, desc="Validating", unit="batch"):
            features, gt = features.to(device), gt.to(device)
            outputs = rnn_model(features)
            loss = criterion(outputs, gt.view(-1, 1))
            losses.append(loss.item())
    return np.mean(losses)


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    features_dir = './features/'
    gt_dir = './processed/'
    batch_size = 1
    learning_rate = 1e-3
    num_epochs = 10
    max_channels = 768

    dataset = RPpgDataset(features_dir, gt_dir, max_channels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    input_dim = 7 * 7 * max_channels
    rnn_model = RNNModel(input_dim=input_dim, hidden_dim=512, output_dim=1, num_layers=2).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_model(train_dataloader, rnn_model, criterion, optimizer, device)
        val_loss = validate_model(val_dataloader, rnn_model, criterion, device)
        print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")


if __name__ == "__main__":
    main()
