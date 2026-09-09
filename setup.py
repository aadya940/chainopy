from setuptools import setup, find_packages

setup(
    name="chainopy",
    version="1.0.3",
    description="A Python Library for Markov Chain based Stochastic Analysis!",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Aadya Aneesh Chinubhai",
    author_email="aadyachinubhai@gmail.com",
    url="https://github.com/aadya940/chainopy",
    packages=find_packages(),
    include_package_data=True,
)
