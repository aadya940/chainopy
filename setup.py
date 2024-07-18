from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize


def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


_install_requires = parse_requirements("requirements.txt")
_tests_require = parse_requirements("requirements_test.txt")
_docs_require = parse_requirements("requirements_doc.txt")

extensions = [
    Extension(
        name="chainopy._backend._absorbing",
        sources=["chainopy/_backend/_absorbing.pyx"],
    ),
    Extension(
        name="chainopy._backend._is_communicating",
        sources=["chainopy/_backend/_is_communicating.pyx"],
    ),
    Extension(
        name="chainopy._backend._learn_matrix",
        sources=["chainopy/_backend/_learn_matrix.pyx"],
    ),
    Extension(
        name="chainopy._backend._simulate", sources=["chainopy/_backend/_simulate.pyx"]
    ),
    Extension(
        name="chainopy._backend._stationary_dist",
        sources=["chainopy/_backend/_stationary_dist.pyx"],
    ),
]

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="chainopy",
    version="1.0",
    packages=["chainopy"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aadya940/chainopy",
    author="Aadya Aneesh Chinubhai",
    author_email="aadyachinubhai@gmail.com",
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
    license="LICENSE",
    install_requires=[
        "cython",
        "numpy",
        "scipy",
        "xarray",
        "matplotlib",
        "networkx",
        "torch",
        "numba",
        "statsmodels",
    ],
    setup_requires=[
        "setuptools",
        "wheel",
    ],
    extras_require={
        "tests": _tests_require,
        "docs": _docs_require,
    },
)
