# PyTorch Network Loader

Allows the easy creation of neural networks in PyTorch using `.json` files.

## Requirements

- Install dependencies:
  `pip install -r requirements.txt`
- PyTorch's dependencies[^1]:  
  NVIDIA GPU with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) ~= v11.6
  [^1]: Only required for use with NVIDIA GPU

## How to Use

The following imports will be required: `from netloader.network import Network, load_network`.  
A `.json` file with the desired network architecture will also be required.

### Constructing the `.json` Architecture

The file is structured as a dictionary containing two sub-dictionaries:
- `net`: Parameters for the network with two options: float `dropout_prob`, which is the dropout
  probability, and boolean `2d`, which sets convolutional layers to assume 2D inputs.
- `layers`: List of dictionaries containing information on each layer, where the first layer takes
  the input, and the last layer produces the output

#### Layer Compatibilities

`reshape` layers are required for adding or removing the channels dimension, C, for compatibilities
with linear layers, which require input $N\times L$, where N is the batch size and L is the length
of the input.  
Other layers, such as recurrent, require the dimension C and 1D data, so their inputs must have
shape $N\times C\times L$.  
Some layers, such as convolutional, require dimension C, but can take either 1D
(with argument `2d` = False) or 2D data (with argument `2d` = True), so the inputs would have shape
$N\times C\times L$ or $N\times C\times H\times W$, respectively, where H is the height and W is
the width.

To add a channels dimension, use the layer `squeeze` with the parameters `squeeze` = False and
`dim` = 1.  
To remove the channels dimension if there is only one channel, use the layer `squeeze` with the
parameters `squeeze` = True and `dim` = 1, otherwise, use `reshape` with `output` containing one -1
to merge the two dimensions (i.e. [-1] for 1D data).


#### Layer Types

`layers` have several options, each with its own options:

**Linear layers**
- `linear`: Linear with SELU:
  - `factor`: optional float, _output size_ = `factor` × _network output size_,
    will be used if provided else `features` will be used
  - `features`: optional integer, output size, won't be used if `factor` is provided
  - `dropout`: boolean = False, probability equals `dropout_prob`
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if a SELU activation should be used
- `upsample`: Linear interpolation scales layer input by two
- `sample`: Predicts the mean and standard deviation of a Gaussian distribution
  and randomly samples from it for a variational autoencoder
  - `factor`: optional float, _output size_ = `factor` × _network output size_,
    will be used if provided else `features` will be used
  - `features`: optional integer, output size, won't be used if `factor` is provided

**Convolutional layers**
- `convolutional`: Convolution with padding using replicate and ELU:
  - `filters`: integer, number of convolutional filters
  - `2d`: boolean = False, if input data is 2D
  - `dropout`: boolean = True, probability equals `dropout_prob`
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `kernel`: integer = 3, kernel size
  - `stride`: integer = 1, stride of the kernel
  - `padding`: integer or string = 'same',
    input padding, can an integer or _same_ where _same_ preserves the input shape
- `conv_upscale`: Scales the layer input by two using convolution and pixel shuffle,
  uses stride of 1, same padding and no dropout, uses ELU
  - `filters`: integer, number of convolutional filters
  - `2d`: boolean = False, if input data is 2D
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = true, ELU activation
  - `kernel`: integer = 3, kernel size
- `conv_transpose`: Scales the layer input by two using transpose convolution,
  uses kernel size of 2 and stride 2, uses ELU
  - `filters`: integer, number of convolutional filters
  - `dropout`: boolean = True, probability equals `dropout_prob`
  - `2d`: boolean = False, if input data is 2D
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
- `conv_depth_downscale`: Reduces C to one, uses kernel size of 1, same padding and ELU
  - `2d`: boolean = False, if input data is 2D
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
- `conv_downscale`: Downscales the layer input by two through strided convolution,
  uses kernel size of 2, padding of 1 using replicate and stride 2, uses ELU
  - `filters`: integer, number of convolutional filters
  - `2d`: boolean = False, if input data is 2D
  - `dropout`: boolean = True, probability equals `dropout_prob`
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
- `pool`: Downscales the layer input by two using max pooling
  - `2d`: boolean = False, if input data is 2D

**Recurrent layers**
- `recurrent`: Recurrent layer with ELU:
  - `dropout`: boolean = True, probability equals `dropout_prob`, requires `layers` > 1
  - `batch_norm`: boolean = False, if batch normalisation should be used
  - `activation`: boolean = True, if an ELU activation should be used
  - `layers`: integer = 2, number of GRU layers
  - `filters`: integer = 1, number of output filters;
  - `method`: string = 'gru', type of recurrent layer, can be _gru_, _lstm_ or _rnn_
  - `bidirectional`: string = None,
    if a bidirectional GRU should be used and the method for combining the two directions,
    can be _sum_, _mean_ or _concatenation_

**Utility layers**
- `reshape`: Reshapes the dimensions
  - `output`: tuple[integer] or tuple[integer, integer], output dimensions of input tensor, ignoring
    the first dimension (N) and subsequent dimensions if the number of dimensions in output
    is less than the dimensions of the input tensor, if output = -1, then last two dimensions are
    flattened
- `squeeze`: Adds or removes a dimension
  - `squeeze`: boolean, if dimension should be removed (True) or added (False)
  - `dim`: integer, which dimension should be edited
- `extract`: Extracts values from the previous layer to pass to the output
  - `number`: integer, number of values to extract from the previous layer
- `clone`: Clones a number of features from the previous layer
  - `number`: integer, number of values to clone from the previous layer
- `index`: Slices the output from the previous layer
  - `number`: integer, index slice number
  - `greater`: boolean = True, if slice should be values greater or less than _number_
- `concatenate`: Concatenates the previous layer with a specified layer
  - `layer`: integer, layer index to concatenate the previous layer with
- `shortcut`: Adds the previous layer with the specified layer
  - `layer`: integer, layer index to add to the previous layer
- `skip`: Passes the output from `layer` into the next layer
  - `layer`: integer, layer index to get the output from

### Loading & Using the Network

The following steps import the architecture into PyTorch:

1. Create a network object by calling `Network` with the arguments: _in\_size_, _out\_size_,
   _learning\_rate_, _name_, & _config\_dir_.
2. If loading a network with existing weights, call the function `load_network` with the arguments:
   _load\_num_, _states\_dir_, & _network_,
   this will return the number of epochs the network has been trained for, the network with loaded
   weights, and a list of training and validation losses.
3. To use the network object, such as for training or evaluation, call the network with the argument
   _x_, and the network will return the forward pass.

The network object has attributes:
- `name`: string, name of the network configuration file (without extension)
- `layers`: list\[dictionary\], layers with layer parameters
- `kl_loss`: Tensor, KL divergence loss on the latent space, if using a `sample` layer
- `clone`: Tensor, cloned values from the network if using a `clone` layer
- `extraction`: Tensor, values extracted from the network if using an `extraction` layer
- `network`: ModuleList, network layers
- `optimiser`: Optimizer, optimiser for the network
- `scheduler`: ReduceLROnPlateau, scheduler for the optimiser
- `latent_mse_weight`: float = 1e-2, relative weight if performing an MSE loss on the latent space
- `kl_loss_weight`: float = 1e-1,
  relative weight if performing a KL divergence loss on the latent space
- `extraction_loss`: float = 1e-1, relative weight if performing a loss on the extracted features

**Example `decoder.json`**
```json
{
  "net": {
    "dropout_prob": 0.1,
    "2d": False
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

from netloader.network import Network, load_network

decoder = Network(5, 240, 1e-5, 'decoder', '../network_configs/')
initial_epoch, decoder, (train_loss, val_loss) = load_network(1, '../model_states/', decoder)

x = torch.rand((10, 5))
output = decoder(x)
```
