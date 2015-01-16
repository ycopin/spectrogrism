#from distutils.core import setup
from setuptools import setup # required for 'install_requires'

setup(
    name='spectrogrism',
    description='Grism-based spectrograph modeling',
    version='0.1',
    py_modules=['spectrogrism'],
    url='https://github.com/ycopin/spectrogrism',
    license='LGPL v3.0',
    author='Yannick Copin',
    author_email='y.copin@ipnl.in2p3.fr',
    requires=['numpy', 'matplotlib'],
    install_requires=['numpy', 'matplotlib'],
)
