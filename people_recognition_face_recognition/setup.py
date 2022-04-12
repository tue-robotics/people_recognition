#!/usr/bin/env python

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['people_recognition_face_recognition'],
    package_dir={'': 'src'},
)

setup(**d)


#     package_data={'': ['*net.pt']},

#     install_requires=[
#         'numpy',
#         'requests',
#         'torchvision',
#         'pillow',
#     ],
