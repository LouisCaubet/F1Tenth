import torch
from torch import nn
from torch.utils.data import DataLoader

import time

import lidar_dataset


print("Running on device: " + 'cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network(nn.Module):

    def __init__(self, lidar_size: int, history_length: int, num_actions: int):
        super(Network, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(lidar_size, 16, 4, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_output_shape = 106

        self.dense = nn.Sequential(
            nn.Linear(conv_output_shape, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        [lidar, velocities] = x
        y = self.conv(lidar)
        y = torch.cat([y, velocities], dim=1)
        y = self.dense(y)
        return y

    def train(self, batch_size, learning_rate, epochs, dataset):
        size = len(dataset)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        # For distributed learning, multiply lr by number of GPUs
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            print("Training: epoch ", epoch, "/", epochs)
            for batchID, (X, vel, y) in enumerate(dataloader):
                X = X.to(device)
                vel = vel.to(device)
                y = y.to(device)
                pred = self([X, vel])
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batchID % 100 == 0:
                    loss, current = loss.item(), batchID * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def count_of_approx_correct(self, prediction, label):
        c = 0
        # Left instead of slightly left
        c += ((prediction==6) * (label==4)).sum().item()
        # Slightly left instead of left
        c += ((prediction==4) * (label==6)).sum().item()
        # Right instead of slightly right
        c += ((prediction==3) * (label==5)).sum().item()
        # Slightly right instead of right
        c += ((prediction==5) * (label==3)).sum().item()
        # Slightly right instead of forward
        c += ((prediction==5) * (label==0)).sum().item()
        # Slightly left instead of forward
        c += ((prediction==4) * (label==0)).sum().item()
        # Forward instead of slightly right
        c += ((prediction==0) * (label==5)).sum().item()
        # Forward instead of slightly left
        c += ((prediction==0) * (label==4)).sum().item()

        return c

    def test_loop(self, test_dataset, loss_fn):
        dataloader = DataLoader(test_dataset, 64, shuffle=True)
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0
        correct = 0
        approx_correct = 0
        total = 0

        with torch.no_grad():
            for X, vel, y in dataloader:
                X = X.to(device)
                vel = vel.to(device)
                y = y.to(device)
                pred = self([X, vel])
                test_loss += loss_fn(pred, y).item()
                _, prediction = torch.max(pred, 1)
                _, label = torch.max(y, 1)
                total += y.size(0)
                correct += (prediction == label).sum().item()

                # Approx correct if "next to" the target (eg. left instead of slightly left)
                approx_correct += self.count_of_approx_correct(prediction, label)


        test_loss /= num_batches
        accuracy = correct / total
        approx_accuracy = accuracy + approx_correct / total

        print("Test results:")
        print("Accuracy: ", accuracy)
        print("Approx. accuracy: ", approx_accuracy)
        print("Average loss: ", test_loss)

        return test_loss


model = Network(809, 10, 8)
model = model.to(device)

print("Starting training...")
start = time.time()

model.train(64, 0.001, 5, lidar_dataset.dataset_train)

print("Training complete!")
print("Training time: " + str(time.time() - start))


print("Starting test phase...")
start = time.time()
avgloss = model.test_loop(lidar_dataset.dataset_test, nn.MSELoss())

print("Testing complete!")
print("Test time: " + str(time.time() - start))

print("Saving model...")
torch.save(model.state_dict(), 'model_weights_' + str(avgloss) + '.pth')
print("Model saved!")
