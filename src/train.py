import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from load import FrameDataset
from features import load_model, extract_features
from flows import CouplingFlow, ResNetFlow
from torchvision import transforms

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


def train_flow_model(flow_model, train_loader, epochs, learning_rate):
    optimizer = optim.Adam(flow_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    flow_model.train()
    for epoch in range(epochs):
        for frames, gt in train_loader:
            frames = frames.to(device)
            gt = gt.to(device)
            features = extract_features(frames, flow_model)

            optimizer.zero_grad()
            outputs = flow_model(features, gt)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


def main():
    data_dir = './processed/'
    model_name = 'swin_base_patch4_window7_224'
    model = load_model(model_name)

    train_dataset = FrameDataset(data_dir, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    flow_model = CouplingFlow(dim=512, n_layers=3, hidden_dims=[256, 256], time_net='TimeFourier', time_hidden_dim=128)
    flow_model.to(device)

    train_flow_model(flow_model, train_loader, epochs=20, learning_rate=0.001)


if __name__ == "__main__":
    main()
