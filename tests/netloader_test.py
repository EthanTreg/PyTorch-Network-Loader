"""
Test to check that NetLoader is working properly
"""
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

from netloader.network import Network
from netloader.networks import Encoder
from netloader.utils.utils import get_device


class TestDataset(Dataset):
    """
    Fake dataset to test netloader.networks
    """
    def __init__(self, in_shape):
        self._in_shape = in_shape
        self._device = get_device()[1]

    def __len__(self):
        return 60

    def __getitem__(self, item):
        target = torch.randint(0, 10, size=(1, 1)).to(self._device).float()
        in_tensor = (torch.ones(size=(1, *self._in_shape[1:])).to(self._device) *
                     target[..., None, None])
        return 0, target[0], in_tensor[0]


def main():
    """
    Main function to test NetLoader
    """
    device = get_device()[1]
    in_shape = [60, 2, 120, 120]
    in_tensor = torch.empty(in_shape).to(device)
    target = torch.zeros([in_shape[0], 10]).to(device)

    # Test all layers
    print('Testing layers...')
    network = Network(
        'layer_examples',
        '../network_configs/',
        [list(in_tensor.shape)[1:], list(target.shape)[1:]],
        list(target.shape)[1:],
        learning_rate=1e-5,
    ).to(device)
    network.group = 1

    out_tensor = network([in_tensor, target])
    loss = torch.nn.MSELoss()(out_tensor, target)
    loss.backward()

    print(f'Output: {out_tensor.shape}')
    print(f'Checkpoints: {len(network.checkpoints)}')

    # Test InceptionV4
    print('Testing InceptionV4...')
    network = Network(
        'inceptionv4',
        '../network_configs/',
        list(in_tensor.shape)[1:],
        list(target.shape)[1:],
        learning_rate=1e-5,
    ).to(device)

    out_tensor = network(in_tensor)
    loss = torch.nn.MSELoss()(out_tensor, target)
    loss.backward()

    # Test Networks
    print('Testing Networks...')
    net = Encoder(0, '', network, verbose='progress', classes=torch.arange(10))
    loader = DataLoader(TestDataset(in_shape), batch_size=60, shuffle=False)
    net.training(5, (loader, loader))

    # Test training
    print('Testing Network training...')
    network = Network(
        'training_test',
        '../network_configs/',
        in_shape[1:],
        list(target.shape)[1:],
        learning_rate=1e-4,
    ).to(device)

    with torch.set_grad_enabled(True):
        for i in range(20):
            losses = []

            for _ in range(10):
                labels = torch.randint(0, 10, size=(in_shape[0], 1)).to(device).float()
                in_tensor = torch.ones(size=in_shape).to(device) * \
                            labels.flatten()[:, None, None, None]
                out_tensor = network(in_tensor)
                loss = nn.CrossEntropyLoss()(out_tensor, labels.flatten().long())
                network.optimiser.zero_grad()
                loss.backward()
                network.optimiser.step()
                losses.append(loss.item())
                network.scheduler.step(losses[-1])

            print(f'Epoch: {i}\tLoss: {np.mean(losses):.2f}')


if __name__ == '__main__':
    main()
