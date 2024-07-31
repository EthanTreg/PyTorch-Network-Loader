"""
Test to check that NetLoader is working properly
"""
import torch
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
        return 600

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
    # networks = ['layer_examples', 'inceptionv4', 'convnext']
    networks = ['layer_examples']

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
    loader = DataLoader(TestDataset(in_shape), batch_size=60, shuffle=False)
    try:
        print(f'Testing {net_name} training...')
        net = Encoder(
            0,
            '',
            network,
            learning_rate=1e-4,
            verbose='full',
            classes=torch.arange(10),
        )
        net.training(1, (loader, loader))
    except NameError:
        pass

    # Test training
    print('Testing Network training...')
    network = Network(
        'training_test',
        '../network_configs/',
        in_shape[1:],
        list(target.shape)[1:],
    ).to(device)

    net = Encoder(
        0,
        '',
        network,
        learning_rate=1e-4,
        verbose='epoch',
        classes=torch.arange(10),
    )
    net.training(20, (loader, loader))
    net.predict(loader)


if __name__ == '__main__':
    main()
