# PyTorch Network Loader

Allows the easy creation and training of neural networks in PyTorch using `.json` files.
Network creation automatically tracks layer output shapes; therefore, knowledge of the input into each layer is not needed.
Networks are loaded from `.json` files, constructed, then a network object is returned that has all the training functionality built into it.

See the [Wiki](../../wiki) for more information on how to use this package.  
For a real-world example of the `Network` object from this package in use, see 
[Fast Spectra Predictor Network](https://github.com/EthanTreg/Spectrum-Machine-Learning).

## Requirements

### Using Within Projects

- Add `netloader @ git+https://github.com/EthanTreg/PyTorch-Network-Loader@LATEST-VERSION`<sup>1</sup> to
  `requirements.txt`
- Install using `pip install -r requirements.txt`
- Example of [InceptionV4](https://arxiv.org/abs/1602.07261) can be downloaded under
  `./network_configs/inceptionv4.json` along with the composite layers in
  `./network_configs/composite_layers/`

<sup>1</sup><sub>To use normalizing flows, `netloader` must be pip installed with the optional argument `flows`: `pip install netloader[flows] @ git+https://github.com/EthanTreg/PyTorch-Network-Loader@LATEST-VERSION`</sub>  

### Locally Running NetLoader

- Clone or download the repository
- Install dependencies:
  `pip install -r requirements.txt`
- PyTorch's dependencies[^1]:  
  NVIDIA GPU with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) ~= v12.1
  [^1]: Only required for use with NVIDIA GPU, v11.8 is also supported, but requirements.txt will
  try to install the v12.1 version