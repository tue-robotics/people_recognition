#!/usr/bin/env python

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['people_recognition_2d'],
    package_dir={'': 'src'},
)

setup(**d)
