#!/usr/bin/env python3

import sys, os

# Try using setuptools first, if it's installed
from setuptools import setup, find_packages

# Figure out conway pattern files to include
pattern_files = []
for root, dirs, files in os.walk("patterns"):
    for file in files:
        pattern_files.append(os.path.join(root,file))

# Need to add all dependencies to setup as we go!
setup(name='conway',
      packages=find_packages(),
      version='0.1',
      description='Conway: play Conway\'s game of life in python',
      author='Michael J. Harms',
      author_email='harmsm@gmail.com',
      url='https://github.com/harmsm/conway',
      download_url='https://XX',
      zip_safe=False,
      install_requires=["scipy","numpy","matplotlib"],
      classifiers=[],
      package_data={'': pattern_files})

