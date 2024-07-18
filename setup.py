from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy

extensions = [
    Extension(
        "chainopy._backend._absorbing",
        sources=["chainopy/_backend/_absorbing.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "chainopy._backend._is_communicating",
        sources=["chainopy/_backend/_is_communicating.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "chainopy._backend._learn_matrix",
        sources=["chainopy/_backend/_learn_matrix.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "chainopy._backend._simulate",
        sources=["chainopy/_backend/_simulate.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "chainopy._backend._stationary_dist",
        sources=["chainopy/_backend/_stationary_dist.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["lapack", "blas"],
    ),
]


class CustomBuildExtCommand(build_ext):
    def build_extensions(self):
        super().build_extensions()


setup(
    name="chainopy",
    version="1.0.0",
    description="A Python Library for Markov Chain based Stochastic Analysis!",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aadya Aneesh Chinubhai",
    author_email="aadyachinubhai@gmail.com",
    url="https://github.com/aadya940/chainopy",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions,
    cmdclass={"build_ext": CustomBuildExtCommand},
    install_requires=[
        "Cython",
        "numpy",
        "scipy",
        "xarray",
        "matplotlib",
        "networkx",
        "torch",
        "numba",
        "statsmodels",
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
