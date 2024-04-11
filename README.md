# PyTorch Network Loader

Allows the easy creation of neural networks in PyTorch using `.json` files and automatic tracking of
output shapes knowledge of the input into each layer is not needed.

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
  - `checkpoints`: boolean = False, if checkpoints should be exclusively used, otherwise, the output
    from every layer will be cached, more user-friendly, but larger memory consumption
  - dictionary, default values for specific layers that override the natural default values of those
    layers, the dictionary contains sub-dictionaries named _layer_name_, which contain the
    parameters found the section [Layer Types](#Layer-Types) with the corresponding layer name
- `layers`: List of dictionaries containing information on each layer, where the first layer takes
  the input, and the last layer produces the output

Examples of the layers can be found under the section [Layer Types](#Layer-Types) and an example of creating
the `.json` file can be found in the section
[Loading & Using the Network](#Loading--Using-the-Network) with more examples in the directory
`network_configs`.

#### Layer Compatibilities

Linear layers can take inputs of either $(N,L)$ or $(N,C,\ldots)$, where $N$ is the
batch size, $C$ is the channels and $L$ is the length of the input, if the input dimension is more
than 2, then the layer will try to preserve the number of channels if the number of features is
divisible; otherwise the output will have the shape $(N,1,L)$.

Other layers, such as recurrent, require the dimension "C", so the input dimensions must be
$(N,C,L)$.  

Some layers, such as convolutional, require dimension $C$, but can take 1D, 2D, & 3D inputs, so the
inputs would have shapes $(N,C,L)$, $(N,C,H,W)$, or $(N,C,D,H,W)$, respectively, where $D$ is the
depth, $H$ is the height, $W$ is the width, and $L$ is the length for 1D data.

The `reshape` layer can be used to change the shape of the inputs for compatibility between the
layers.

### Loading & Using the Network

The following steps import the architecture into PyTorch:

1. Create a network object by calling `Network` with the arguments: _in\_shape_, _out\_shape_,
   _learning\_rate_, _name_, & _config\_dir_.
2. To use the network object, such as for training or evaluation, call the network with the argument
   _x_, and the network will return the forward pass.

**All shapes given to the network or layers should exclude the batch dimension $N$.**

The network object has attributes:
- `name`: string, name of the network configuration file (without extension)
- `shapes`: list\[list\[integer\]\], layer output shapes
- `layers`: list\[dictionary\], layers with layer parameters
- `checkpoints`: list\[Tensor\], cloned values from the network's `checkpoint` layers
- `network`: ModuleList, network layers
- `optimiser`: Optimizer, optimiser for the network
- `scheduler`: ReduceLROnPlateau, scheduler for the optimiser
- `layer_num`: integer = None, number of layers to use, if None use all layers
- `group`: integer = 0, which group is the active group if a layer has the group attribute
- `kl_loss_weight`: float = 1e-1,
  relative weight if performing a KL divergence loss on the latent space
- `kl_loss`: Tensor = 0, KL divergence loss on the latent space, if using a `sample` layer

**Example `decoder.json`**

```json
{
  "net": {
    "checkpoints": false,
    "linear": {
      "dropout": 0.1
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
      "factor": 1,
      "activation": false
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

`layers` have several options, each with its own parameters.

All layers can take the optional `group` parameter which means that that layer will only be active
if the network attribute `group` is equal to the layer's `group`.  
This is most useful if the head gets changed during training.  
`skip` layers should be used between groups so that the expected input shape is correct.  
See `layer_examples.json` under `network_configs` to see how to use groups and other layers.

**Linear layers**
  in an autoencoder to encode the most important information in the first values of the latent space
- `Linear`: Linear/fully connected
  - `features`: optional integer, output size, won't be used if `factor` is provided
  - `factor`: optional float, _features_ = `factor`$\times$_network output size_,
    will be used if provided else `features` will be used
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if a SELU activation should be used
  - `dropout`: float = 0.01, probability of dropout
- `OrderedBottleneck`: Information-ordered bottleneck to randomly change the size of the bottleneck
  - `min_size`: integer = 0, minimum gate size
- `Sample`: Gets the mean and standard deviation of a Gaussian distribution from $C$ in the previous
  layer, halving $C$, and randomly samples from it, mainly for a variational autoencoder
- `Upsample`: Linear interpolation to scale the layer input
  - `shape`: list\[integer\] = None, shape of the output, will be used if provided, else factor will
    be used
  - `scale`: list\[float\] = 2, factor to upscale all or individual dimensions, first dimension is
    ignored, won't be used if out_shape is provided
  - `mode`: {'linear', 'nearest', 'bilinear', 'bicubic', 'trilinear'}, what interpolation method to
    use for upsampling

**Convolutional layers**
- `AdaptivePool`: Uses pooling to downscale the layer input to the desired shape
  - `shape`: integer | list\[integer\], output shape of the layer
  - `channels`: boolean = True, if the input includes a channels dimension
  - `mode`: {'average', 'max'}, whether to use 'max' or 'average' pooling
- `Conv`: Convolution with padding using replicate and ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `stride`: integer | list\[integer\] = 1, stride of the kernel
  - `kernel`: integer | list\[integer\] = 3, kernel size
  - `padding`: integer | string | list\[integer\] = 'same',
    input padding, can an integer, list of integers or _same_ where _same_ preserves the input shape
  - `dropout`: float = 0.1, probability of dropout
- `ConvDepthDownscale`: Reduces $C$ to one, uses kernel size of 1, same padding and ELU
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `dropout`: float = 0, probability of dropout
- `ConvDownscale`: Downscales the layer input using strided convolution
  uses kernel size of 2, padding of 1 using replicate and stride 2, uses ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `scale`: integer = 2, stride and size of the kernel, which acts as the downscaling factor
  - `dropout`: float = 0.1, probability of dropout
- `ConvTranspose`: Scales the layer input using transpose convolution
  uses kernel size of 2 and stride 2, uses ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `scale`: integer = 2, stride and size of the kernel, which acts as the upscaling factor
  - `out_padding`: integer = 0, padding applied to the output
  - `dropout`: float = 0.1, probability of dropout
- `ConvUpscale`: Scales the layer input using convolution and pixel shuffle
  uses stride of 1, same padding and no dropout, uses ELU
  - `filters`: optional integer, number of convolutional filters, will be used if provided, 
    else `factor` will be used
  - `factor`: optional float, _filters_ = `factor`$\times$_network output channels_, 
    won't be used if `filters` is provided
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, ELU activation
  - `scale`: integer = 2, factor to upscale the input by
  - `kernel`: integer | list\[integer\] = 3, kernel size
  - `dropout`: float = 0.1, probability of dropout
- `PixelShuffle`: Equivalent to torch.nn.PixelShuffle, but for N-dimensions
  - `scale`: integer, upscaling factor
- `Pool`: Performs pooling
  - `kernel`: integer | list\[integer\] = 2, size of the kernel 
  - `stride`: integer | list\[integer\] = 2, stride of the kernel 
  - `padding`: integer | string = 0, input padding, can an integer or 'same' where 'same' preserves
    the input shape
  - `mode`: {'max', 'average'}, whether to use 'max' or 'average' pooling
- `PoolDownscale`: Downscales the input using pooling
  - `scale`: integer, stride and size of the kernel, which acts as the downscaling factor
  - `mode`: string = 'max', whether to use max pooling ('max') or average pooling ('average')

**Recurrent layers**
- `Recurrent`: Recurrent layer with ELU
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `layers`: integer = 2, number of recurrent layers
  - `filters`: integer = 1, number of output filters;
  - `dropout`: float = -1, probability of dropout, requires `layers` > 1
  - `method`: {'gru', 'rnn', 'lstm'}, type of recurrent layer
  - `bidirectional`: {None, 'sum', 'mean', 'concatenate'}, if a bidirectional GRU should be used and
    the method for combining the two directions

**Utility layers**
- `Checkpoint`: Saves the output from the previous layer for use in future layers
- `Concatenate`: Concatenates the previous layer with a specified layer
  - `layer`: integer, layer index to concatenate the previous layer with
  - `checkpoint`: boolean = False, if `layer` should be relative to checkpoints or network layers,
    if `checkpoints` in `net` is True, `layer` will always be relative to checkpoints
  - `dim` : integer = 0, dimension to concatenate to (not including $N$)
- `Index`: Slices the output from the previous layer
  - `number`: integer, index slice number
  - `greater`: boolean = True, if slice should be values greater or less than _number_
- `Reshape`: Reshapes the dimensions
  - `shape`: list\[integer\], output dimensions of input tensor, ignoring the first dimension ($N$)
- `Shortcut`: Adds the previous layer with the specified layer
  - `layer`: integer, layer index to add to the previous layer
  - `checkpoint`: boolean = False, if `layer` should be relative to checkpoints or network layers,
    if `checkpoints` in `net` is True, `layer` will always be relative to checkpoints
- `Skip`: Passes the output from `layer` into the next layer
  - `layer`: integer, layer index to get the output from
  - `checkpoint`: boolean = False, if `layer` should be relative to checkpoints or network layers,
    if `checkpoints` in `net` is True, `layer` will always be relative to checkpoints

**Composite layers**  
Custom blocks can be made from the layers above and inserted into the network.
This is useful if making repetitive blocks such as the Inception block
([Szegedy, et al. 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)).  
The block is created in the same way a network is as a `.json` file.  
In the `network.json` file, the block can be inserted by creating a `composite` layer with
parameters:
- `Composite`: Custom layer that combines multiple layers in a `.json` file for repetitive use
  - `name`: string, name of the subnetwork
  - `config_dir`: string, path to the directory with the network configuration file
  - `checkpoint`: boolean = False, if layer index should be relative to checkpoint layers
  - `channels`: optional integer, number of output channels, won't be used if `shape` is
    provided, if `channels` and `shape` aren't provided, the input dimensions will be preserved
  - `shape`: optional list\[integer\], output shape of the block, will be used if provided;
    otherwise, `channels` will be used
  - `defaults`: optional dictionary, default values for the parameters for each type of layer, same
    as for the dictionary parameter for `net`
