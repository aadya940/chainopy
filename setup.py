from setuptools import setup, Extension
import numpy as np
import os

os.chdir(".")
os.system("cython --version")

os.system("cython chainopy/_backend/_absorbing.pyx")
os.system("cython chainopy/_backend/_is_communicating.pyx")
os.system("cython chainopy/_backend/_learn_matrix.pyx")
os.system("cython chainopy/_backend/_simulate.pyx")
os.system("cython chainopy/_backend/_stationary_dist.pyx")


def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


_install_requires = parse_requirements("requirements.txt")
_tests_require = parse_requirements("requirements_test.txt")
_docs_require = parse_requirements("requirements_doc.txt")

extensions = [
    Extension(
        name="chainopy._backend._absorbing",
        sources=["chainopy/_backend/_absorbing.c"],
    ),
    Extension(
        name="chainopy._backend._is_communicating",
        sources=["chainopy/_backend/_is_communicating.c"],
    ),
    Extension(
        name="chainopy._backend._learn_matrix",
        sources=["chainopy/_backend/_learn_matrix.c"],
    ),
    Extension(
        name="chainopy._backend._simulate", sources=["chainopy/_backend/_simulate.c"]
    ),
    Extension(
        name="chainopy._backend._stationary_dist",
        sources=["chainopy/_backend/_stationary_dist.c"],
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
    ext_modules=extensions,
    include_dirs=[np.get_include()],
    license="LICENSE",
    install_requires=["Cython"] + _install_requires,
    setup_requires=[
        "Cython",
    ],
    extras_require={
        "tests": _tests_require,
        "docs": _docs_require,
    },
)
