import torch
import torch.nn as nn
from torch.utils import data
import os
import random
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
from prettytable import PrettyTable

from dataset import utils, pure, ubfc_rppg
from evaluate import metric, postprocess
from configs import running
from models import loss_function
from . import cnn, transformer


def merge_clips(x):
    sort_x = sorted(x.items(), key=lambda x: x[0])
    sort_x = [i[1] for i in sort_x]
    sort_x = np.concatenate(sort_x, axis=0)
    return sort_x.reshape(-1)


def train_test(path, train_config, test_config, mode="Train", model_path=""):
    train_set = utils.MyDataset(train_config)
    test_set = utils.MyDataset(test_config)
    train_iter = data.DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True)
    test_iter = data.DataLoader(test_set, batch_size=test_config.batch_size, shuffle=False)
    net = cnn.EfficientPhys(frame_depth=train_config.frame_depth, img_size=train_config.H)

    if mode == "Train":
        net = net.to(train_config.device)
        lr = train_config.lr
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0)

        print(f"START TRAINING!")
        table_config = PrettyTable()
        table_config.field_names = ["Model", "Batch Size", "Learning Rate", "Epochs", "Frame Depth"]
        table_config.add_row(
            ["EfficientPhys", train_config.batch_size, lr, train_config.num_epochs, train_config.frame_depth])
        print(table_config)

        train(net, optimizer, train_iter, train_config, test_iter, test_config, path)
        torch.save(net.state_dict(), path + os.sep + f"trans_efficient.pt")
    else:
        assert model_path, "Pretrained model is required!"
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)

    net = net.to(test_config.device)
    print(f"EVLUATING: {test_config.batch_size} BATCH SIZE")
    temp = test(net, test_iter, test_config)
    table = PrettyTable()
    table.field_names = ["MAE", "RMSE", "MAPE", "R"]
    table.add_row(temp)
    print(table)


def train(net: nn.Module, optimizer: torch.optim.Optimizer,
          train_iter: data.DataLoader, train_config: running.TrainEfficient,
          test_iter: data.DataLoader, test_config: running.TestEfficient, path):
    os.makedirs(path, exist_ok=True)
    net = net.to(train_config.device)
    net.train()
    loss_fun = loss_function.NegPearson()
    train_loss = metric.Accumulate(1)
    base_len = train_config.num_gpu * train_config.frame_depth
    all_test = []

    for epoch in range(train_config.num_epochs):
        net.train()
        train_loss.reset()
        with tqdm(total=len(train_iter), desc=f"TRAINING # EPOCH {epoch + 1}") as pbar:
            for x, y, _, _, _ in train_iter:
                x = x.to(train_config.device).permute(0, 2, 1, 3, 4)
                y = y.to(train_config.device)
                B, T, C, H, W = x.shape
                x = x.reshape(-1, C, H, W)
                y = y.reshape(-1, 1)
                x = x[: B * T // base_len * base_len]
                y = y[: B * T // base_len * base_len]
                last_frame = x[-1, :, :, :].unsqueeze(0).repeat(train_config.num_gpu, 1, 1, 1)
                x = torch.cat([x, last_frame], 0)

                preds = net(x)
                preds = preds.reshape(B, T)
                y = y.reshape(B, T)

                loss = loss_fun(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.update(val=loss.data, n=1)
                pbar.update(1)

        torch.save(net.state_dict(), path + os.sep + f"efficient_epoch{epoch + 1}.pt")
        # print(f"EPOCH {epoch + 1}: Train loss: {train_loss.acc[0] / train_loss.cnt[0]: .3f}")
        train_result = PrettyTable()
        train_result.field_names = ["AT EPOCH", "TRAIN LOSS"]
        train_result.add_row([epoch + 1, f"{train_loss.acc[0] / train_loss.cnt[0]: .3f}"])
        print(train_result)
        table = PrettyTable()
        table.field_names = ["MAE", "RMSE", "MAPE", "R"]
        with tqdm(total=len(test_iter), desc=f"TESTING # EPOCH {epoch + 1}") as pbar:
            temp = test(net, test_iter, test_config, pbar)
            # print(f"MAE: {temp[0]: .3f}, RMSE: {temp[1]: .3f}, MAPE: {temp[2]: .3f}, R: {temp[3]: .3f}")
            table.add_row(temp)
        print(table)
        # all_test.append(temp)

    # for i in range(train_config.num_epochs):
    # print(all_test[i])


def test(net: nn.Module, test_iter: data.DataLoader,
         test_config: running.TestEfficient, pbar=None) -> list:
    net = net.to(test_config.device)
    net.eval()
    predictions = dict()
    labels = dict()
    base_len = test_config.num_gpu * test_config.frame_depth

    for x, y, _, subjects, chunks in test_iter:
        x = x.to(test_config.device).permute(0, 2, 1, 3, 4)
        y = y.to(test_config.device)

        B, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        y = y.reshape(-1, 1)
        x = x[: B * T // base_len * base_len]
        y = y[: B * T // base_len * base_len]
        last_frame = x[-1, :, :, :].unsqueeze(0).repeat(test_config.num_gpu, 1, 1, 1)
        x = torch.cat([x, last_frame], 0)
        preds = net(x)

        for i in range(B):
            file_name = subjects[i]
            chunk_idx = chunks[i]
            if file_name not in predictions.keys():
                predictions[file_name] = dict()
                labels[file_name] = dict()
            predictions[file_name][chunk_idx] = preds[i * T: (i + 1) * T].detach().cpu().numpy()
            labels[file_name][chunk_idx] = y[i * T: (i + 1) * T].detach().cpu().numpy()
        if pbar:
            pbar.update(1)

    pred_phys = []
    label_phys = []
    for file_name in predictions.keys():
        pred_temp = merge_clips(predictions[file_name])
        label_temp = merge_clips(labels[file_name])
        if test_config.post == "fft":
            pred_temp = postprocess.fft_physiology(pred_temp, Fs=float(test_config.Fs),
                                                   diff=test_config.diff,
                                                   detrend_flag=test_config.detrend).reshape(-1)
            label_temp = postprocess.fft_physiology(label_temp, Fs=float(test_config.Fs),
                                                    diff=test_config.diff,
                                                    detrend_flag=test_config.detrend).reshape(-1)
        else:
            pred_temp = postprocess.peak_physiology(pred_temp, Fs=float(test_config.Fs),
                                                    diff=test_config.diff,
                                                    detrend_flag=test_config.detrend).reshape(-1)
            label_temp = postprocess.peak_physiology(label_temp, Fs=float(test_config.Fs),
                                                     diff=test_config.diff,
                                                     detrend_flag=test_config.detrend).reshape(-1)
        pred_phys.append(pred_temp)
        label_phys.append(label_temp)
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)

    return metric.cal_metric(pred_phys, label_phys)
