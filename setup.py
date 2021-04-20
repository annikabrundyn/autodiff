#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='autodiff',
    version='0.0.0',
    description='Basic reverse mode automatic differentiation library',
    author='Annika Brundyn',
    author_email='ab8690@nyu.edu',
    url='https://github.com/annikabrundyn/autodiff',
    package_dir={"": "autodiff"},
    packages=find_packages("autodiff"),
)