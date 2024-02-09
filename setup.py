from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(name="chainopy._backend._absorbing", sources=["chainopy/_backend/_absorbing.pyx"]),
    Extension(
        name="chainopy._backend._is_communicating",
        sources=["chainopy/_backend/_is_communicating.pyx"],
    ),
    Extension(
        name="chainopy._backend._learn_matrix", sources=["chainopy/_backend/_learn_matrix.pyx"]
    ),
    Extension(name="chainopy._backend._simulate", sources=["chainopy/_backend/_simulate.pyx"]),
    Extension(
        name="chainopy._backend._stationary_dist",
        sources=["chainopy/_backend/_stationary_dist.pyx"],
    ),
]


setup(
    name="chainopy",
    version="1.0",
    packages=["chainopy"],
    author="Aadya Aneesh Chinubhai",
    author_email="aadyachinubhai@gmail.com",
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
)
