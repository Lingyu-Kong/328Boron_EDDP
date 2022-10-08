from setuptools import setup
from Cython.Build import cythonize
from distutils.core import setup, Extension
import numpy

# package = Extension('_threebody_indices', ['_threebody_indices.pyx'], include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize([package]))
package = Extension('feature.eddp_feature', ['feature/eddp_feature.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))

# package1 = Extension('feature', ['eddp_feature.pyx'], include_dirs=[numpy.get_include()])
# setup(ext_modules=cythonize([package, package1]))

## usage
# python setup.py build_ext --inplace