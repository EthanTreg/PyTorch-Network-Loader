"""
Package setup
"""
from setuptools import setup

import netloader

setup(
    name='netloader',
    version=netloader.__version__,
    description='Utility to generate and train PyTorch neural network objects from JSON files',
    url='https://github.com/EthanTreg/PyTorch-Network-Loader',
    author='Ethan Tregidga',
    author_email='ethan.tregidga@epfl.ch',
    license='MIT',
    packages=[
        'netloader',
        'netloader.layers',
        'netloader.utils',
        'netloader.networks',
        'netloader.models',
    ],
    install_requires=['numpy', 'torch'],
    extras_require={
        'flows': ['zuko']
    }
)
