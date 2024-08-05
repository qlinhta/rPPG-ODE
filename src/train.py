import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt


class RPPGDataset(Dataset):
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
            features = np.pad(features, ((0, 0), (0, pad_size)), mode='constant')

        gt_value = gt
        return torch.tensor(features, dtype=torch.float32), torch.tensor(gt_value, dtype=torch.float32)


def collate_fn(batch):
    features, gt = zip(*batch)
    features = torch.stack(features)
    gt = torch.stack(gt)
    return features, gt


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out.squeeze(-1)


class DenseModel(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        x = x.view(batch_size * seq_len, num_features)  # Flatten the input to (batch_size * seq_len, num_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(batch_size, seq_len)  # Reshape to (batch_size, seq_len)
        return x


def train_model(dataloader, model, criterion, optimizer, device):
    model.train()
    losses = []
    progress = tqdm(dataloader, desc="Training", unit="batch", ascii=True)
    running_loss = 0.0
    for i, (features, gt) in enumerate(progress):
        features, gt = features.to(device), gt.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=running_loss / (i + 1))
        losses.append(loss.item())
    return np.mean(losses)


def validate_model(dataloader, model, criterion, device):
    model.eval()
    losses = []
    running_loss = 0.0
    progress = tqdm(dataloader, desc="Validating", unit="batch", ascii=True)
    with torch.no_grad():
        for i, (features, gt) in enumerate(progress):
            features, gt = features.to(device), gt.to(device)
            outputs = model(features)
            loss = criterion(outputs, gt)

            running_loss += loss.item()
            losses.append(loss.item())
            progress.set_postfix(loss=running_loss / (i + 1))
    return np.mean(losses)


def test_model(dataloader, model, device):
    model.eval()
    predictions = []
    ground_truths = []
    subjects = []
    with torch.no_grad():
        for features, gt in tqdm(dataloader, desc="Testing", unit="batch", ascii=True):
            features, gt = features.to(device), gt.to(device)
            outputs = model(features)
            predictions.append(outputs.cpu().numpy())
            ground_truths.append(gt.cpu().numpy())
            subjects.append(features.cpu().numpy())

    return predictions, ground_truths


def plot_results(predictions, ground_truths):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for i in range(min(4, len(predictions))):
        row, col = i // 2, i % 2
        ax[row, col].plot(ground_truths[i].flatten(), label="Ground Truth", color='blue', linewidth=1.5)
        ax[row, col].plot(predictions[i].flatten(), label="Prediction", color='red', linewidth=1.5)
        ax[row, col].minorticks_on()
        ax[row, col].grid(which='major', linestyle='-', linewidth='0.5', color='grey', zorder=0, alpha=0.3)
        ax[row, col].grid(which='minor', linestyle='-', linewidth='0.5', color='gray', zorder=0, alpha=0.2)
        ax[row, col].legend()
    plt.tight_layout()
    plt.savefig(f'./figures/rppg_signal_comparison.pdf', bbox_inches='tight')
    plt.show()


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FEATURE = "efficientnet_b0"
    FEATURE_DIR = f'./features/{FEATURE}'
    GROUND_TRUTH = './processed/'
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-1
    EPOCHS = 10
    MAX_CHANNELS = 1

    table = PrettyTable()
    table.field_names = ["Feature Extractor", "Batch Size", "Learning Rate", "Num Epochs", "Max Channels"]
    table.add_row([FEATURE, BATCH_SIZE, LEARNING_RATE, EPOCHS, MAX_CHANNELS])
    print(table)

    dataset = RPPGDataset(FEATURE_DIR, GROUND_TRUTH, MAX_CHANNELS)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    input_dim = MAX_CHANNELS
    SEQ_LEN = 250
    # lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=512, output_dim=1, num_layers=2).to(DEVICE)
    model = DenseModel(input_dim=input_dim, seq_len=SEQ_LEN, output_dim=1).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_model(train_dataloader, model, criterion, optimizer, DEVICE)
        val_loss = validate_model(val_dataloader, model, criterion, DEVICE)
        table = PrettyTable()
        table.field_names = ["Epoch", "Training Loss", "Validation Loss"]
        table.add_row([epoch + 1, train_loss, val_loss])
        print(table)

    predictions, ground_truths = test_model(test_dataloader, model, DEVICE)
    plot_results(predictions, ground_truths)

    os.makedirs("./saved_models", exist_ok=True)
    torch.save(model.state_dict(),
               f"./saved_models/lstm_model_{FEATURE}_{MAX_CHANNELS}_{EPOCHS}_{LEARNING_RATE}.pt")


if __name__ == "__main__":
    main()
