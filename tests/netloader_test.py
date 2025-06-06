"""
Test to check that NetLoader is working properly
"""
import os

import torch
import numpy as np

import netloader.networks as nets
from netloader import transforms
from netloader.network import Network
from netloader.utils.utils import get_device
from netloader.data import BaseDataset, loader_init


class Dataset(BaseDataset):
    """
    Mock dataset with input values equal to target label
    """
    def __init__(self, samples: int, shape: list[int]) -> None:
        super().__init__()
        self.low_dim = np.random.random((samples, 1))
        self.high_dim = np.ones((samples, *shape)) * self.low_dim[:, 0, *[np.newaxis] * len(shape)]


def main():
    """
    Main function to test NetLoader
    """
    device = get_device()[1]
    in_shape = [60, 1, 100, 100]
    in_tensor = torch.empty(in_shape).to(device)
    target = torch.zeros([in_shape[0], 10]).to(device)
    networks = ['layer_examples', 'inceptionv4', 'convnext']
    states_dir = './model_states/'

    if not os.path.exists(states_dir):
        os.mkdir(states_dir)

    for net_name in networks:
        print(f'Testing {net_name}...')
        net_in_shape = [
            in_shape[1:],
            list(target.shape)[1:],
        ] if net_name == 'layer_examples' else in_shape[1:]

        network = Network(
            net_name,
            '../network_configs/',
            net_in_shape,
            list(target.shape)[1:],
        ).to(device)

        if net_name == 'layer_examples':
            network.group = 1
            out_tensor = network([in_tensor, target])
        else:
            out_tensor = network(in_tensor)

        loss = torch.nn.MSELoss()(out_tensor, target)
        loss.backward()

    # Initialise datasets
    dataset = Dataset(1000, [1, 100, 100])
    high_transform = transforms.MultiTransform(
        transforms.Normalise(data=dataset.high_dim),
        transforms.NumpyTensor(),
    )
    low_transform = transforms.MultiTransform(
        transforms.Normalise(data=dataset.low_dim, mean=False),
        transforms.NumpyTensor(),
    )
    dataset.high_dim = high_transform(dataset.high_dim)
    dataset.low_dim = low_transform(dataset.low_dim)
    loaders = loader_init(dataset, ratios=(0.8, 0.2))

    # Test Networks
    try:
        print(f'Testing {net_name} training...')
        net = nets.Encoder(
            0,
            '',
            network,
            learning_rate=1e-4,
            verbose='full',
            classes=torch.arange(10),
        ).to(device)
        net.training(1, loaders)
    except NameError:
        pass

    # Test training
    print('Testing Network training...')
    network = Network(
        'test_encoder',
        '../network_configs/',
        in_shape[1:],
        [1],
    ).to(device)

    net = nets.Encoder(
        1,
        states_dir,
        network,
        overwrite=True,
        learning_rate=1e-4,
        verbose='epoch',
        transform=low_transform,
        in_transform=high_transform,
    ).to(device)
    net.training(1, loaders)
    net = nets.load_net(1, states_dir, net.net.name).to(device)
    net.training(10, loaders)
    net.predict(loaders[1])

    # Test architectures
    network = Network(
        'test_decoder',
        '../network_configs/',
        [1],
        in_shape[1:],
    ).to(device)
    net = nets.Decoder(
        1,
        states_dir,
        network,
        overwrite=True,
        learning_rate=1e-4,
        verbose='epoch',
        transform=low_transform,
        in_transform=high_transform,
    ).to(device)
    net.training(1, loaders)
    net = nets.load_net(1, states_dir, net.net.name)
    net.training(2, loaders)
    net.predict(loaders[1])

    print('TEST COMPLETE')


if __name__ == '__main__':
    main()
