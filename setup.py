# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pos_ngrams',
    version='0.1.0',
    description='Transform text into POS tuples and generate ngrams',
    long_description=readme,
    author='Peter J Usherwood',
    author_email='peterjusherwood93@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

