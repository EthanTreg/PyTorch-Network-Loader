# PyTorch Network Loader

Allows the easy creation of neural networks in PyTorch using `.json` files.

## Requirements

### Using Within Projects

- Add `netloader @ git+https://github.com/EthanTreg/PyTorch-Network-Loader@LATEST-VERSION` to
  `requirements.txt`
- Install using `pip install -r requirements.txt`
- Example of [InceptionV4](https://arxiv.org/abs/1602.07261) can be downloaded under
  `./network_configs/inceptionv4.json` along with the composite layers in
  `./network_configs/composite_layers/`

### Locally Running NetLoader

- Clone or download the repository
- Install dependencies:
  `pip install -r requirements.txt`
- PyTorch's dependencies[^1]:  
  NVIDIA GPU with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) ~= v11.6
  [^1]: Only required for use with NVIDIA GPU

## How to Use

The following import is required: `from netloader.network import Network`.  
A `.json` file with the desired network architecture is also required.

### Constructing the `.json` Architecture

The file is structured as a dictionary containing two sub-dictionaries:
- `net`: Global network parameters with the following options:
  - `2d`: boolean, if layers that can accept 2D and 1D inputs should assume 2D inputs
  - `checkpoints`: boolean = False, if checkpoints should be exclusively used, otherwise, the output
    from every layer will be cached, more user-friendly, but larger memory consumption
  - dictionary, default values for each layer that override the values in `default.py`,
    the dictionary contains sub-dictionaries named _layer_name_, which contain the parameters
    found the section [Layer Types](#Layer-Types) with the corresponding layer name
- `layers`: List of dictionaries containing information on each layer, where the first layer takes
  the input, and the last layer produces the output

Examples of the layers can be found under the section [Layer Types](#Layer-Types) and an example of creating
the `.json` file can be found in the section
[Loading & Using the Network](#Loading--Using-the-Network) with more examples in the directory
`network_configs`.

#### Layer Compatibilities

Linear layers can take inputs of either $N\times L$ or $N\times C\times L$, where $N$ is the
batch size, $C$ is the channels and $L$ is the length of the input.  
Other layers, such as recurrent, require the dimension $C$ and 1D data, so their inputs must have
shape $N\times C\times L$.  
Some layers, such as convolutional, require dimension $C$, but can take either 1D
(with argument `2d` = false) or 2D data (with argument `2d` = true), so the inputs would have shape
$N\times C\times L$ or $N\times C\times H\times W$, respectively, where $H$ is the height and $W$ is
the width.
The `reshape` layer can be used to change the shape of the inputs
for compatibility between the layers.

### Loading & Using the Network

The following steps import the architecture into PyTorch:

1. Create a network object by calling `Network` with the arguments: _in\_shape_, _out\_shape_,
   _learning\_rate_, _name_, & _config\_dir_.
2. To use the network object, such as for training or evaluation, call the network with the argument
   _x_, and the network will return the forward pass.

The network object has attributes:
- `name`: string, name of the network configuration file (without extension)
- `layers`: list\[dictionary\], layers with layer parameters
- `checkpoints`: list\[Tensor\], cloned values from the network's `checkpoint` layers
- `kl_loss`: Tensor, KL divergence loss on the latent space, if using a `sample` layer
- `network`: ModuleList, network layers
- `optimiser`: Optimizer, optimiser for the network
- `scheduler`: ReduceLROnPlateau, scheduler for the optimiser
- `layer_num`: integer = None, number of layers to use, if None use all layers
- `latent_mse_weight`: float = 1e-2, relative weight if performing an MSE loss on the latent space
- `kl_loss_weight`: float = 1e-1,
  relative weight if performing a KL divergence loss on the latent space

**Example `decoder.json`**

```json
{
  "net": {
    "2d": false,
    "linear": {
      "dropout": 1
    }
  },
  "layers": [
    {
      "type": "linear",
      "features": 120
    },
    {
      "type": "linear",
      "features": 120
    },
    {
      "type": "linear",
      "factor": 1
    }
  ]
}
```

**Example code**
```python
import torch

from netloader.network import Network

decoder = Network([5], [240], 1e-5, 'decoder', '../network_configs/')

x = torch.rand((10, 5))
output = decoder(x)
```

### Layer Types

`layers` have several options, each with its own options:

**Linear layers**
- `ordered_bottleneck`: Information-ordered bottleneck to randomly change the size of the bottleneck
  in an autoencoder to encode the most important information in the first values of the latent space
- `linear`: Linear with SELU
  - `features`: optional integer, output size, won't be used if `factor` is provided
  - `factor`: optional float, _features_ = `factor`$\times$_network output size_,
    will be used if provided else `features` will be used
  - `dropout`: float = 0.01, dropout probability
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if a SELU activation should be used
- `sample`: Gets the mean and standard deviation of a Gaussian distribution from $C$ in the previous
  layer, halving $C$, and randomly samples from it for a variational autoencoder
- `upsample`: Linear interpolation scales layer input by two

**Convolutional layers**
- `adaptive_pool`: Uses average pooling to downscale the input to the desired shape
  - `output`: list[integer], output shape of the layer ignoring $C$
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
- `convolutional`: Convolution with padding using replicate and ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
  - `dropout`: float = 0.1, dropout probability
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `kernel`: integer | list[integer] = 3, kernel size
  - `stride`: integer = 1, stride of the kernel
  - `padding`: integer | list[integer] | string = 'same',
    input padding, can an integer, list of integers or _same_ where _same_ preserves the input shape
- `conv_depth_downscale`: Reduces $C$ to one, uses kernel size of 1, same padding and ELU
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
- `conv_downscale`: Downscales the layer input by two through strided convolution,
  uses kernel size of 2, padding of 1 using replicate and stride 2, uses ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
  - `dropout`: float = 0.1, dropout probability
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
- `conv_transpose`: Scales the layer input by two using transpose convolution,
  uses kernel size of 2 and stride 2, uses ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
  - `dropout`: float = 0.1, dropout probability
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `out_padding`: integer = 0, padding applied to the output
- `conv_upscale`: Scales the layer input by two using convolution and pixel shuffle,
  uses stride of 1, same padding and no dropout, uses ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, ELU activation
  - `kernel`: integer | list[integer] = 3, kernel size
- `pool`: Performs max pooling
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
  - `kernel`: integer = 2, size of the kernel 
  - `stride`: integer = 2, stride of the kernel 
  - `padding`: integer | string = 0, input padding, can an integer or 'same'
    where 'same' preserves the input shape;
  - `mode`: string = 'max', whether to use max pooling ('max') or average pooling ('average')
- `pool_downscale`: Downscales the input layer by `factor` using max pooling
  - `factor`: float, factor to downscale the input
  - `2d`: optional boolean, if input data is 2D, if not provided, `2d` from `net` is used
  - `mode`: string = 'max', whether to use max pooling ('max') or average pooling ('average')

**Recurrent layers**
- `recurrent`: Recurrent layer with ELU
  - `dropout_prob`: float = -1, dropout probability, if < 0, `dropout_prob` from `net` is used,
    requires `layers` > 1
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `layers`: integer = 2, number of GRU layers
  - `filters`: integer = 1, number of output filters;
  - `method`: string = 'gru', type of recurrent layer, can be _gru_, _lstm_ or _rnn_
  - `bidirectional`: string = None,
    if a bidirectional GRU should be used and the method for combining the two directions,
    can be _sum_, _mean_ or _concatenation_

**Utility layers**
- `checkpoint`: Saves the output from the previous layer for use in future layers
- `concatenate`: Concatenates the previous layer with a specified layer
  - `layer`: integer, layer index to concatenate the previous layer with
  - `checkpoint`: boolean = False, if `layer` should be relative to checkpoints or network layers,
    if `checkpoints` in `net` is True, `layer` will always be relative to checkpoints
  - `dim` : integer = 0, dimension to concatenate to (not including $N$)
- `index`: Slices the output from the previous layer
  - `number`: integer, index slice number
  - `greater`: boolean = True, if slice should be values greater or less than _number_
- `reshape`: Reshapes the dimensions
  - `output`: list[integer], output dimensions of input tensor, ignoring the first dimension ($N$)
- `shortcut`: Adds the previous layer with the specified layer
  - `layer`: integer, layer index to add to the previous layer
  - `checkpoint`: boolean = False, if `layer` should be relative to checkpoints or network layers,
    if `checkpoints` in `net` is True, `layer` will always be relative to checkpoints
- `skip`: Passes the output from `layer` into the next layer
  - `layer`: integer, layer index to get the output from
  - `checkpoint`: boolean = False, if `layer` should be relative to checkpoints or network layers,
    if `checkpoints` in `net` is True, `layer` will always be relative to checkpoints

**Composite layers**  
Custom blocks can be made from the layers above and inserted into the network.
This is useful if making repetitive blocks such as the Inception block
([Szegedy, et al. 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)).  
The block is created in the same way a network is as a `.json` file.  
In the network.json file, the block can be inserted by creating a `composite` layer with parameters:
- `Composite`: Custom layer that combines multiple layers in a `.json` file for repetitive use
  - `config_path`: string, path to the `.json` file containing the block architecture
  - `channels`: optional integer, number of output channels, won't be used if `out_shape` is provided,
    if `channels` and `out_shape` aren't provided, the input dimensions will be preserved
  - `out_shape`: optional list[intger], output shape of the block, will be used if provided;
    otherwise, `channels` will be used
  - `defaults`: optional dictionary, default values for each layer that override the network's
    default values, the dictionary contains sub-dictionaries named layer_name, which contain the
    parameters found in the layers above with the corresponding layer name
