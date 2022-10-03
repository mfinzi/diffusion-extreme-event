from setuptools import setup,find_packages
import sys, os

setup(name="diffusion-extreme-events",
      description="Event conditioning with diffusion models",
      version='0.1',
      author='Marc Finzi',
      author_email='maf820@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py','tables','flax','optax','clu','chex'],#jax, 
      packages=find_packages(),
      long_description=open('README.md').read(),
)