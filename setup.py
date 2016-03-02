#!/usr/bin/env python
# Time-stamp: <2016-03-01 10:39:54 ycopin>

from setuptools import setup

name = 'spectrogrism'
version = '0.6'

cmdclass = {}
command_options = {}
# Add 'sphinx' command in case sphinx is installed
try:
    from sphinx.setup_command import BuildDoc
    cmdclass.update({'sphinx': BuildDoc})
    command_options.update({'sphinx': {
        'version': ('setup.py', version),
        'release': ('setup.py', version)}
    })
except ImportError:
    pass

setup(
    name=name,
    version=version,
    description='Grism-based spectrograph modeling',
    url='https://github.com/ycopin/spectrogrism',
    author='Yannick Copin',
    author_email='y.copin@ipnl.in2p3.fr',
    license='LGPL v3.0',
    packages=['spectrogrism'],
    # Data
    package_data={'spectrogrism': ['data/*']},
    # Setup extensions
    cmdclass=cmdclass,
    command_options=command_options,
)
