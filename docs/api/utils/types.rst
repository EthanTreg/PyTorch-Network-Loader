.. _api_utils_types:

.. module:: netloader.utils.types
   :synopsis: Type aliases and variables for NetLoader.

Types
=====

.. currentmodule:: netloader.utils.types

Type Aliases
------------

.. py:data:: Param

   Alias of :class:`dict`\[:class:`str`, :class:`~torch.Tensor` \| :data:`~netloader.utils.types.Param`\]

.. autocleantype:: DataLike

.. autocleantype:: ShapeLike

.. autocleantype:: ArrayLike

.. autocleantype:: TensorLike

.. autocleantype:: NDArrayLike

.. autocleantype:: DataListLike

.. autocleantype:: TensorListLike

.. autocleantype:: NDArrayListLike

Type Variables
--------------

.. autocleantype:: P

.. autocleantype:: T

.. autocleantype:: DataT

.. autocleantype:: ArrayT

.. autocleantype:: ModuleT

.. autocleantype:: TensorT

.. autocleantype:: NDArrayT

.. autocleantype:: DatasetT

.. autocleantype:: DataListT

.. autocleantype:: TensorListT

.. autocleantype:: ModuleListT

.. autocleantype:: ArrayCT

.. autocleantype:: LossCT

.. autocleantype:: TensorLossCT

Protocols
---------

.. autoclass:: DatasetProtocol
   :special-members: __len__,__getitem__
