"""
Test to check that NetLoader is working properly
"""
import torch
import numpy as np
from torch import nn

from netloader.network import Network
from netloader.utils.utils import get_device


def main():
    """
    Main function to test NetLoader
    """
    device = get_device()[1]
    in_shape = [60, 2, 120, 120]
    in_tensor = torch.empty(in_shape).to(device)
    target = torch.zeros([in_shape[0], 10]).to(device)

    # Test all layers
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

    # Test training
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
