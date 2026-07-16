# PyTorch Network Loader

PyTorch Network Loader is a utility that enables the easy creation of neural networks in PyTorch using JSON
configuration files.
The package automatically tracks layer output shapes, eliminating the need to manually calculate input dimensions for
each layer.
NetLoader also provides an architecture framework that allows for the easy architecture setup and training framework,
along with additional useful utilities to accelerate the use of neural networks in science.

See the [Wiki](https://pytorch-network-loader.readthedocs.io) for more information on how to use this package.  
For a real-world example of this package, see [Bayesian-DARKSKIES](https://github.com/EthanTreg/Bayesian-DARKSKIES).

## Requirements

### Using Within Projects

- Install using `pip install netloader`[^1]
- Example of [InceptionV4](https://arxiv.org/abs/1602.07261) can be downloaded under
  `./network_configs/inceptionv4.json` along with the composite layers in
  `./network_configs/composite_layers/`

[^1]: To use normalising flows or weights & biases, `netloader` must be pip installed with the optional argument
`flows` and/or `wandb`:
`pip install netloader[flows,wandb]`

### Locally Running NetLoader

- Clone or download the repository
- Install dependencies:
  `pip install -r requirements.txt`
- PyTorch's dependencies[^2]:  
  NVIDIA GPU with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) ~= v12.1
  [^2]: Only required for use with NVIDIA GPU, v11.8 is also supported, but requirements.txt will
  try to install the v12.1 version
