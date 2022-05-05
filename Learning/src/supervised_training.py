import torch
from torch import nn
from torch.utils.data import DataLoader
import time
from lidar_dataset import LidarDataset

import lidar_dataset
from network import Network

from config_loader import *


def train_model(model: Network, batch_size: int, learning_rate: float, epochs: int, dataset: LidarDataset):
    size = len(dataset)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print("Training: epoch ", epoch, "/", epochs)
        for batchID, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batchID % 100 == 0:
                loss, current = loss.item(), batchID * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(model: Network, test_dataset: LidarDataset, loss_fn):
    dataloader = DataLoader(test_dataset, 64, shuffle=True)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    precision_angle = 0
    precision_velocity = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            precision_angle += torch.sum(torch.abs(pred[:, 0] - y[:, 0]))
            precision_velocity += torch.sum(
                torch.abs(pred[:, 1] - y[:, 1]))

    test_loss /= num_batches
    precision_angle /= size
    precision_velocity /= size

    print("Test results:")
    print("Steering angle precision: ", precision_angle)
    print("Velocity precision: ", precision_velocity)
    print("Average loss: ", test_loss)

    return test_loss


print("Running on device: " + 'cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Network()
model = model.to(device)

if RESUME_TRAINING:
    model.load_state_dict(torch.load(os.path.join(
        os.path.dirname(__file__), DEFAULT_PTH_FILE), map_location=device))

print("Starting training...")
start = time.time()

train_model(model, 64, 0.001, 5, lidar_dataset.dataset_train)

print("Training complete!")
print("Training time: " + str(time.time() - start))


print("Starting test phase...")
start = time.time()
avgloss = test_loop(model, lidar_dataset.dataset_test, nn.MSELoss())

print("Testing complete!")
print("Test time: " + str(time.time() - start))

print("Saving model...")
torch.save(model.state_dict(), 'model_weights_' + str(avgloss) + '.pth')
print("Model saved!")
