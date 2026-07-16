.. _api_data:

.. module:: netloader.data
   :synopsis: Data and data loading utilities for NetLoader.

Data
====

.. currentmodule:: netloader.data

Classes
-------

.. autoclass:: Data
   :special-members: __len__,__getitem__

.. autoclass:: DataList
   :special-members: __add__,__len__,__getitem__,__iter__

.. autoclass:: BaseDataset
   :special-members: __len__,__getitem__

Functions
---------

.. autofunction:: data_collation

.. autofunction:: loader_init

Types
-----

.. autocleantype:: InArrayT_contra

.. autocleantype:: OutArrayT_co

Protocols
---------

.. autoclass:: ApplyFunc
   :special-members: __call__
