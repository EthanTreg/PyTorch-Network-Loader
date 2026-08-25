.. _api_data_structures:

.. module:: netloader.data.structures
   :synopsis: Data structures for use with NetLoader.

Structures
==========

.. currentmodule:: netloader.data.structures

Classes
-------

.. autoclass:: Data
   :special-members: __eq__,__len__,__getitem__

.. autoclass:: DataList
   :special-members: __add__,__eq__,__len__,__getitem__,__iter__

Functions
---------

.. autofunction:: data_collation

Types
-----

.. autocleantype:: InArrayT_contra

.. autocleantype:: OutArrayT_co

Protocols
---------

.. autoclass:: ApplyFunc
   :special-members: __call__
