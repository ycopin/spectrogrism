# http://stackoverflow.com/questions/2058802
# import pkg_resources            # part of setuptools
# __version__ = pkg_resources.require("spectrogrism")[0].version
# del pkg_resources

__version__ = '0.6'

from . import spectrogrism
from . import distortion
from . import snifs
from . import nisp
