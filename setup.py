"""
Package setup
"""
from setuptools import setup

setup(
    name='netloader',
    version='3.0.0',
    description='Utility to generate and train PyTorch neural network objects from JSON files',
    url='https://github.com/EthanTreg/PyTorch-Network-Loader',
    author='Ethan Tregidga',
    author_email='ethan.tregidga@epfl.ch',
    license='MIT',
    packages=['netloader', 'netloader.layers', 'netloader.utils', 'netloader.networks'],
    install_requires=['numpy', 'torch', 'pandas'],
    extras_require={
        'flows': ['zuko']
    }
)
