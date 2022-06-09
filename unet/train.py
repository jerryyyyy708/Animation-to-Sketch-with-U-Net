import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import SketchUNet
from dataset import SketchDataset


def train(limit_epoch, limit_size, data_path, sketch_path, output_path):
    if torch.cuda.is_available():  
        torch.cuda.empty_cache()
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    train_dataset = SketchDataset(
        data_path=data_path, 
        sketch_path=sketch_path, 
        limit_size=limit_size
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True,
    )

    model = SketchUNet().to(dev)
    # print(summary(model, (3, 512, 512)))

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.L1Loss()
    train_loss_plt = []
    best_lost = 1e9
    for epoch in range(limit_epoch):
        print(f"Epoch: {epoch}")

        model.train()
        train_loss = []
        
        for i, (data, sketch) in enumerate(train_dataloader):
            data, sketch = data.to(dev), sketch.to(dev)
            optimizer.zero_grad()
            sketch_hat = model(data)
            train_batch_loss = criterion(sketch_hat, sketch)
            train_batch_loss.backward()
            optimizer.step()

            train_loss.append(train_batch_loss.item())
            print(f"Batch [{i + 1} / {len(train_dataloader)}] with loss {train_batch_loss.item()}.", end="\r")
        
        cur_train_loss = sum(train_loss) / len(train_loss)
        train_loss_plt.append(cur_train_loss)
        print(f"Train loss: {cur_train_loss}")
        
        if epoch < 10 or (1 + epoch) % 50 == 0:
            print(f"Save Model on epoch {epoch}.")
            torch.save(model.state_dict(), os.path.join(output_path, f"epoch_{str(epoch)}.model"))

        # currently test with training data
        # maybe add valid data test here
        model.eval()
        for i, (data, sketch) in enumerate(train_dataloader):
            data, sketch = data.to(dev), sketch.to(dev)
            if (i + epoch) % 10 != 0:
                continue

            hat = model(data)
            hat_np = (hat.cpu().detach().numpy())
            shape = hat_np.shape
            hat_np = hat_np.reshape(shape[2], shape[3])

            filename = os.path.join(output_path, f"{str(epoch)}_{str(i)}.jpg")
            plt.imsave(filename, hat_np, cmap="gray")
    plt.legend(['Loss']) 
    plt.plot(train_loss_plt)
    plt.show()



train(
    limit_epoch=300, 
    limit_size=384, 
    data_path="data", 
    sketch_path="sketch",
    output_path="result2"
)
