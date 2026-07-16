.. _changelog:

*********
Changelog
*********

Version 3.11.0
==============

**Release Data:** July 2026

**Install:** ``pip install netloader==3.11.0``

**Additions:**

* Added :class:`~netloader.models.misc.MultiNetwork`.
* Added error when importing :class:`~netloader.architectures.flows.NormFlow` and
  :class:`~netloader.architectures.flows.NormFlowEncoder` from :mod:`~netloader.architectures` and
  :class:`~netloader.layers.flows.SplineFlow` without ``Zuko`` package.
* Added RNG state saving to :class:`~netloader.architectures.base.BaseArchitecture`.
* Added ``extra`` argument to ``_loss`` and ``_loss_tensor`` methods of
  :class:`~netloader.architectures.base.BaseArchitecture`.
* Added option for callable functions/objects for
  :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_loss_weights``.
* Added :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_step`` method for each iteration.
* Added support for more schedulers in
  :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_update_scheduler``.
* Added :meth:`~netloader.architectures.base.BaseArchitecture.get_param_groups` method to define optimiser param groups.
* Added :class:`~netloader.architectures.base.BaseArchitecture` save frequency.
* Added ``description`` argument to :class:`~netloader.layers.base.BaseLayer`.
* Added support for multiple layer outputs to :class:`~netloader.layers.base.BaseMultiLayer`.
* Added forward pass to :class:`~netloader.layers.base.BaseMultiLayer` to pack outputs.
* Added :meth:`~netloader.data.DataList.__add__` magic method.
* Added :meth:`~netloader.data.DataList.get_data` method.
* Added :meth:`~netloader.data.BaseDataset.step` method.
* Added :func:`~netloader.data.loader_init` function to initialise data loaders.
* Added ``activation_kwargs`` to :class:`~netloader.layers.linear.Activation`.
* Added :class:`~netloader.loss_funcs.GeneralLoss` for any :mod:`~torch.nn` loss function.
* Added :class:`~netloader.network.BaseNetwork`.
* Added :meth:`~netloader.network.Network.get_config` method.
* Added :class:`~netloader.network.Branch`.
* Added :mod:`~netloader.schedulers`.
* Added :class:`~netloader.transforms.BaseTypeTransform` where input output has same type.
* Added support for loading networks with different weight names if weight shapes match.
* Added :class:`~netloader.network.Network`\ .\ _logger attribute.
* Added :class:`~netloader.network.Network`\ .\ _load_state_dict method.
* Added more type hints.

**Changes:**

* Changed :class:`~netloader.layers.multi_layer.Concatenate`, :class:`~netloader.layers.multi_layer.Pack`,
  :class:`~netloader.layers.multi_layer.Shortcut`, :class:`~netloader.layers.multi_layer.Skip`, &
  :class:`~netloader.layers.multi_layer.Unpack` layers to be under :mod:`~netloader.layers.multi_layer`.
* Changed :class:`~netloader.architectures.base.BaseArchitecture` to only flag overwriting if base name is matching.
* Changed :class:`~netloader.data.Data`\ .\ ``__repr__`` type to give specific type.
* Changed :class:`~netloader.architectures.flows.NormFlowEncoder` to allow different learning rates for normalising flow
  and encoder.
* Changed :class:`~netloader.transforms.BaseTransform` to abstract class.
* Changed :class:`~netloader.transforms.BaseTransform` to allow different input output types.
* Changed :class:`~netloader.utils.utils.safe_globals` to allow only accepting classes defined in ``__all__``.
* Changed :class:`~netloader.architectures.base.BaseArchitecture` to abstract class.
* Changed :class:`~netloader.layers.base.BaseLayer` to abstract class.
* Changed Python version to 3.14.
* Changed package versions.

**Removals:**

* Removed support for networks saved in old deprecated weights format.

**Deprecations:**

* Deprecated ``networks`` for name change to architectures.
* Deprecated ``set_optimiser`` & ``set_scheduler`` for
  :class:`~netloader.architectures.utils.UtilityMixin.init_optimiser` and
  :class:`~netloader.architectures.utils.UtilityMixin.init_scheduler` methods.

**Fixes:**

* Fixed :meth:`~netloader.architectures.base.BaseArchitecture.save` method not recovering on SystemExit.
* Fixed :class:`~netloader.architectures.base.BaseArchitecture` not changing Network device if compiled.
* Fixed :class:`~netloader.utils.configs.BaseConfig`\ .\ ``__setattr__`` using incorrect type hints.
* Fixed back-propagation issue with :class:`~netloader.layers.linear.Sample`.
* Fixed loading RNG states if state was saved on GPU.

Version 3.10.1
==============

**Release Date:** February 2026

**Install:** ``pip install netloader@git+https://github.com/EthanTreg/PyTorch-Network-Loader@2c1c4c3``

**Additions:**

* Added :class:`~netloader.layers.misc.Dropout` layer.
* Added :mod:`~netloader.utils.configs` dataclasses for :class:`~netloader.network.Network` configuration.
* Added :func:`~netloader.utils.utils.compare_versions` for version comparison.
* Added :func:`~netloader.utils.utils.dict_list_append`, :func:`~netloader.utils.utils.dict_list_convert`, and
  :func:`~netloader.utils.utils.list_dict_convert`.
* Added :func:`~netloader.utils.utils.suppress_logger_warnings` wrapper to suppress warnings for a module.
* Added NetLoader version to :class:`~netloader.layers.base.BaseLayer`.
* Added :class:`~netloader.layers.base.BaseLayer`\ .\ ``__getstate__`` magic method.
* Added :meth:`~netloader.architectures.base.BaseArchitecture.compile` method.
* Added :meth:`~netloader.architectures.base.BaseArchitecture.get_losses` method.
* Added loss weights to :class:`~netloader.architectures.base.BaseArchitecture` with ``get`` and ``set``
  ``loss_weights`` methods and automatic calculation of total loss.
* Added :class:`~netloader.models.convnext.ConvNeXt`\ .\ ``_stem`` private method.
* Added ``padding_mode`` keyword argument to convolutional layers.
* Added support for :class:`~netloader.data.DataList` inputs in :func:`~netloader.data.data_collation`.
* Added support for :class:`~torch.nn.ModuleList` with :class:`~netloader.network.CompatibleNetwork`.
* Added ``pool_kwargs`` keyword argument to :class:`~netloader.layers.pooling.Pool`.
* Added ``packaging`` as requirement.
* Added :class:`~netloader.layers.misc.Pad` layer.
* Added :class:`~netloader.architectures.base.BaseArchitecture.get_save_path`.
* Added :meth:`~netloader.data.Data.apply` & :meth:`~netloader.data.DataList.apply` methods.

**Changes:**

* Changed :class:`~netloader.layers.base.BaseMultiLayer`\ .\ ``_target`` attribute to accept ``list[list[int]]`` shapes.
* Changed activation functions to use ``inplace`` where possible by default.
* Changed :class:`~netloader.data.DataList` length to return data length rather than list length.
* Changed :class:`~netloader.architectures.encoder_decoder.Autoencoder` to use a dictionary for loss terms.
* Changed :class:`~netloader.architectures.encoder_decoder.Autoencoder` to use a dictionary for loss weights.
* Changed :class:`~netloader.architectures.flows.NormFlowEncoder` to use a dictionary for loss weights.
* Changed :mod:`~netloader.layers` to be more compatible with :func:`~torch.compile`.
* Changed package versions.

**Removals:**

* Removed deprecated :class:`~netloader.layers.base.BaseLayer`\ .\ ``layers`` attribute.

**Deprecations:**

* Deprecated :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``save_path`` attribute and replaced with
  :meth:`~netloader.architectures.base.BaseArchitecture.set_save_path` method.
* Deprecated :class:`~netloader.network.Network`\ .\ ``config`` attribute and replaced with
  :meth:`~netloader.network.Network.get_config` method.
* Deprecated ``ArrayTC`` and replaced with :data:`~netloader.utils.types.ArrayCT`.
* Deprecated Python versions < 3.12.

**Fixes:**

* Fixed :mod:`~netloader.utils.utils` not accessible from :mod:`~netloader.utils`.
* Fixed batch printing from :class:`~netloader.architectures.base.BaseArchitecture` not using total loss.
* Fixed reduced precision training.
* Fixed total loss for loss functions returning dictionaries without total key.
* Fixed :class:`~netloader.layers.linear.OrderedBottleneck` blocking last feature.
* Fixed :class:`~netloader.layers.misc.Checkpoint` not saving combined outputs.
* Fixed :meth:`~netloader.network.Network.to` method not working for non-tensors in ``checkpoints`` attribute.
* Fixed :class:`~netloader.transforms.Normalise` scale zero division.
* Fixed network name in :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``__repr__`` magic method.
* Fixed formatting of :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``__repr__``.
* Fixed :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_logger`` not initialised from saved state.
* Fixed getting :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``save_path`` deprecation.
* Fixed overwrite errors from :meth:`~netloader.architectures.base.BaseArchitecture.set_save_path` if new save path is
  same as original.
* Fixed same asymmetric padding for convolutional and pooling layers.
* Fixed false deprecation warning in :class:`~netloader.network.Composite` layer
* Fixed incorrect attribute saving for :class:`~netloader.layers.pooling.AdaptivePool`.
* Fixed type hints for :class:`~netloader.utils.utils.Shapes`.

Version 3.9.4
=============

**Release Date:** October 2025

**Install:** ``pip install netloader@git+https://github.com/EthanTreg/PyTorch-Network-Loader@v3.9.4``

**Additions:**

* Added base layers to :mod:`~netloader.layers`.
* Added ``save_freq`` attribute to :class:`~netloader.architectures.base.BaseArchitecture`.
* Added keyword argument check to :class:`~netloader.layers.pooling.AdaptivePool`.
* Added :meth:`~netloader.architectures.base.BaseArchitecture.get_hyperparams` method.

**Changes:**

* Changed :class:`~netloader.network.Network` to not convert to CPU when saving.
* Changed some type hints.

**Deprecations:**

* Deprecated ``ArrayTC`` for :data:`~netloader.utils.types.ArrayCT`.

**Fixed:**

* Fixed support for some old NetLoader versions.
* Fixed save corruption of :class:`~netloader.architectures.base.BaseArchitecture` during keyboard interrupt.
* Fixed keyword argument name clash in :class:`~netloader.layers.misc.Index`.

Version 3.9.1
=============

**Release Date:** October 2025

**Additions:**

* Added :class:`~netloader.layers.base.BaseSingleLayer` class.
* Added :class:`~netloader.utils.utils.Shapes` class.
* Added :class:`~netloader.data.Data` class for data & uncertainties.
* Added :class:`~netloader.data.DataList` class for storing multiple arrays or tensors.
* Added :class:`~netloader.architectures.base.BaseArchitecture` loss dictionary support.
* Added support to :class:`~netloader.architectures.base.BaseArchitecture` for list of transforms.
* Added :class:`~netloader.data.DataList` support to :class:`~netloader.architectures.base.BaseArchitecture`.
* Added :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_loss_tensor`` method.
* Added :class:`~netloader.data.Data` support to :class:`~netloader.transforms.BaseTransform`.
* Added :class:`~netloader.models.convnext.ConvNeXt`\ .\ ``_head`` method.
* Added ``return_idxs`` to :func:`~netloader.data.loader_init`.
* Added :func:`~netloader.data.data_collation` to collate lists of arrays, tensors, :class:`~netloader.data.Data`, and
  :class:`~netloader.data.DataList` objects.
* Added ``loss_func`` argument to :class:`~netloader.architectures.encoder_decoder.BaseEncoder`.
* Added :class:`~netloader.loss_funcs.GaussianNLLLoss` class.
* Added :class:`~netloader.layers.multi_layer.Pack` class.
* Added ``layer`` & ``checkpoint`` arguments to :class:`~netloader.layers.multi_layer.Unpack`.
* Added support for :class:`~netloader.data.DataList` to several layers.
* Added :class:`~netloader.network.CompatibleNetwork` class.
* Added :class:`~netloader.network.Network`\ .\ ``version`` attribute.
* Added ``len``, ``iter``, and ``getitem`` magic methods to :class:`~netloader.network.Network`.
* Added ``label`` keyword argument to :func:`~netloader.utils.utils.ascii_plot`.
* Added several types in :mod:`~netloader.utils.types` module.
* Added :class:`~netloader.layers.misc.BatchNorm` layer.
* Added support to set total loss in :class:`~netloader.architectures.base.BaseArchitecture` if loss is a dictionary.
* Added option to only pack output from previous layer in :class:`~netloader.layers.multi_layer.Pack` layer.
* Added ``root`` keyword argument to :class:`~netloader.network.Network` for loading networks and composite layers.

**Changes:**

* Changed layers to use :class:`~netloader.utils.utils.Shapes` class for layer shapes.
* Changed most methods and functions to not allow positional arguments for keyword arguments.
* Changed :meth:`~netloader.architectures.base.BaseArchitecture.predict` to not return None values.
* Changed :func:`~netloader.architectures.base.load_arch` to accept ``kwargs`` to pass to :func:`~torch.load`.
* Changed :class:`~netloader.layers.blocks.ConvNeXtBlock` to :mod:`~netloader.layers`.
* Changed :class:`~netloader.layers.misc.Index` layer to use indexing or slicing.
* Changed type hints to be more consistent.
* Changed package versions.

**Removals:**

* Removed ``UNSET`` global variable.

**Deprecations:**

* Deprecated ``BaseLayer.layers`` attribute.
* Deprecated ``BaseLayer.forward`` method.
* Deprecated :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_loss_func`` method.
* Deprecated ``number`` and ``greater`` arguments in :class:`~netloader.layers.misc.Index` layer.

**Fixes:**

* Fixed :class:`~netloader.models.convnext.ConvNeXt` weight initialisation for biases that are None.
* Fixed :class:`~netloader.network.Network` ignoring ``layer_num`` attribute.
* Fixed :class:`~netloader.transforms.Log` uncertainty inplace operations.
* Fixed :class:`~netloader.data.BaseDataset` error message for incorrect attribute length.
* Fixed :class:`~netloader.layers.multi_layer.Concatenate` layer when ``dim=-1``.
* Fixed :class:`~netloader.transforms.Index` transform not detecting old state format.
* Fixed :class:`~netloader.transforms.Normalise` transform error for ``offset`` or ``scale`` ndarray keyword arguments.
* Fixed backwards compatibility with Python-3.11.
* Fixed type hint errors.

Version 3.8.0
=============

**Release Date:** August 2025

**Install:** ``pip install netloader@git+https://github.com/EthanTreg/PyTorch-Network-Loader@v3.8.0``

**Additions:**

* Added :class:`~netloader.models.convnext.ConvNeXt` implemented using classes.
* Added ``group`` attribute to :class:`~netloader.layers.base.BaseLayer`.
* Added :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``losses`` functionality to save multiple loss
  values.
* Added print functions to :class:`~netloader.architectures.base.BaseArchitecture` for custom print logic during
  training & predicting.
* Added :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_loss_func`` method to return Tensor before
  :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_loss`` method returns float.
* Added :meth:`~netloader.transforms.MultiTransform.extend` method to add multiple transforms.
* Added ASCII plot for loss curve during training.
* Added NetLoader version to :class:`~netloader.architectures.base.BaseArchitecture` save state.
* Added ``extra_repr`` method to :class:`~netloader.layers.misc.DropPath`.
* Added GitHub reference to timm for DropPath.

**Changes:**

* Changed :func:`~netloader.data.loader_init` ``ratios`` argument to split the dataset after ``idxs``.
* Changed :class:`~netloader.network.Network` forward pass to use :class:`~netloader.layers.base.BaseLayer` rather than
  config for information.
* Changed :class:`~netloader.network.Network` to catch all exceptions during the forward pass.

**Fixes:**

* Fixed :class:`~netloader.layers.misc.DropPath` not working if ``prob`` is 0.
* Fixed :class:`~netloader.architectures.flows.NormFlowEncoder` optimiser initialisation.
* Fixed optimiser and scheduler kwargs not being saved.
* Fixed :class:`~netloader.transforms.Index` weights only safe saving.
* Fixed :class:`~netloader.transforms.Normalise` uncertainty in-place operation.
* Fixed non-NetLoader classes being added to PyTorch safe globals.
* Fixed type hints.

Version 3.6.1
=============

**Release Date:** June 2025

**Install:** ``pip install netloader@git+https://github.com/EthanTreg/PyTorch-Network-Loader@v3.6.1``

**Additions:**

* Added :class:`~netloader.data.BaseDataset` for creating datasets.
* Added ``get_device`` & ``get_epochs`` to :class:`~netloader.architectures.base.BaseArchitecture`.
* Added ``overwrite`` keyword argument to :class:`~netloader.architectures.base.BaseArchitecture` to prevent unwanted
  file deletion.
* Added Jupyter Notebook package use example.

**Changes:**

* Changed ``input_`` keyword argument to inputs in :meth:`~netloader.architectures.base.BaseArchitecture.predict`
  method.
* Changed :class:`~netloader.network.Network` ``in_shape`` and ``out_shape`` arguments to accept tuples.
* Changed :class:`~netloader.transforms.Index` to allow negative ``in_shape`` dimension sizes for undefined lengths.
* Changed package requirements.

**Fixes:**

* Fixed docstrings & type hints.

Earlier Versions
================

See the `GitHub Releases <https://github.com/EthanTreg/PyTorch-Network-Loader/releases>`_
for complete version history.

Upgrade Guide
=============

From 3.9.x to 3.11.x
--------------------

No breaking changes. Existing code should be compatible.
