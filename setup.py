# Author: Anand Patil
# Date: 6 Feb 2009
# License: GPL
####################################

from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('duffy',parent_package=None,top_path=None)

config.add_extension(name='cut_geographic',sources=['duffy/cut_geographic.f'])
config.add_extension(name='postproc_utils',sources=['duffy/postproc_utils.f'])

config.packages = ["duffy"]

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))