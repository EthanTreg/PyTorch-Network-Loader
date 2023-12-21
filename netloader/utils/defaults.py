"""
Default values for layers
"""
DEFAULTS = {
  'checkpoints': False,
  'ordered_bottleneck': {
    'min_size': 0,
  },
  'linear': {
    'batch_norm': False,
    'activation': True,
    'dropout': 0.01,
  },
  'sample': {},
  'upsample': {},
  'adaptive_pool': {},
  'convolutional': {
    'batch_norm': False,
    'activation': True,
    'kernel': 3,
    'stride': 1,
    'dropout': 0.1,
    'padding': 'same',
  },
  'conv_depth_downscale': {
    'batch_norm': False,
    'activation': True,
  },
  'conv_downscale': {
    'batch_norm': False,
    'activation': True,
    'dropout': 0.1,
  },
  'conv_transpose': {
    'batch_norm': False,
    'activation': True,
    'out_padding': 0,
    'dropout': 0.1,
  },
  'conv_upscale': {
    'batch_norm': False,
    'activation': True,
    'kernel': 3,
  },
  'pool': {
    'kernel': 2,
    'stride': 2,
    'padding': 0,
    'mode': 'max',
  },
  'pool_downscale': {
    'mode': 'max',
  },
  'recurrent': {
    'batch_norm': False,
    'activation': True,
    'layers': 2,
    'filters': 1,
    'dropout': 0.1,
    'method': 'gru',
    'bidirectional': None,
  },
  'checkpoint': {},
  'clone': {},
  'concatenate': {
    'dim': 0,
  },
  'extract': {},
  'index': {
    'greater': True,
  },
  'reshape': {},
  'shortcut': {},
  'skip': {},
}
