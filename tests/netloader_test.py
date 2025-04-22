"""
Test to check that NetLoader is working properly
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader

import netloader.networks as nets
from netloader import transforms
from netloader.network import Network
from netloader.utils.utils import get_device


class TestDataset(Dataset):
    """
    Fake dataset to test netloader.networks
    """
    def __init__(self, in_shape: list[int]):
        self._in_shape: list[int] = in_shape
        self.transform: transforms.BaseTransform | None = None

    def __len__(self) -> int:
        return 600

    def __getitem__(self, item: int):
        target: torch.Tensor = torch.randint(0, 10, size=(1, 1)).float()
        in_tensor: torch.Tensor = (torch.ones(size=(1, *self._in_shape[1:])) *
                                   target[..., None, None])

        if self.transform:
            target = self.transform(target)

        return 0, target[0], in_tensor[0]


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

    # Test Networks
    dataset = TestDataset(in_shape)
    loader = DataLoader(dataset, batch_size=60, shuffle=False)
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
        net.training(1, (loader, loader))
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
    dataset.transform = transforms.Normalise(next(iter(loader))[1])

    net = nets.Encoder(
        1,
        states_dir,
        network,
        learning_rate=1e-4,
        verbose='epoch',
        transform=dataset.transform,
    ).to(device)
    net.training(1, (loader, loader))
    net = nets.load_net(1, states_dir, net.net.name).to(device)
    net.training(10, (loader, loader))
    net.predict(loader)

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
        learning_rate=1e-4,
        verbose='epoch',
        in_transform=dataset.transform,
    ).to(device)
    net.training(1, (loader, loader))
    net = nets.load_net(1, states_dir, net.net.name)
    net.training(2, (loader, loader))
    net.predict(loader)


if __name__ == '__main__':
    main()
