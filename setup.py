from spectrogrism import __version__
from setuptools import setup

setup(
    name='spectrogrism',
    description='Grism-based spectrograph modeling',
    version=__version__,
    py_modules=['spectrogrism', 'nisp'],
    url='https://github.com/ycopin/spectrogrism',
    license='LGPL v3.0',
    author='Yannick Copin',
    author_email='y.copin@ipnl.in2p3.fr',
    # requires=['numpy', 'matplotlib'],
    # install_requires=['numpy', 'matplotlib'],
)
