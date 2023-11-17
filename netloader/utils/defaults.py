"""
Default values for layers
"""
DEFAULTS = {
  'linear': {
    'dropout': False,
    'batch_norm': False,
    'activation': True,
  },
  'sample': {},
  'upsample': {},
  'adaptive_pool': {},
  'convolutional': {
    'dropout': True,
    'batch_norm': False,
    'activation': True,
    'kernel': 3,
    'stride': 1,
    'padding': 'same',
  },
  'conv_depth_downscale': {
    'batch_norm': False,
    'activation': True,
  },
  'conv_downscale': {
    'dropout': True,
    'batch_norm': False,
    'activation': True,
  },
  'conv_transpose': {
    'dropout': True,
    'batch_norm': False,
    'activation': True,
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
    'dropout_prob': -1,
    'method': 'gru',
    'bidirectional': None,
  },
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
