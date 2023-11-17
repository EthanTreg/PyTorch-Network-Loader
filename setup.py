"""
Package setup
"""
from setuptools import setup

setup(
    name='netloader',
    version='0.3.0',
    description='Utility to generate PyTorch neural network objects from JSON files',
    url='https://github.com/EthanTreg/PyTorch-Network-Loader',
    author='Ethan Tregidga',
    author_email='et1g19@soton.ac.uk',
    license='MIT',
    packages=['netloader', 'netloader.layers', 'netloader.utils'],
    install_requires=['torch'],
)
