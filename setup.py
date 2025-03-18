from setuptools import setup, find_packages

setup(
    name="labtools",
    version="0.1",
    packages=find_packages(),   # This will find all nested submodules
    install_requires=[]         # Add any core dependencies here if needed
)