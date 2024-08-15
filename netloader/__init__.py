"""
Package information and creates the logger
"""
import logging


__version__ = '3.2.1'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
