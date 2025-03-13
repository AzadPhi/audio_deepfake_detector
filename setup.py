# setup.py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]


setup(name='audio_deepfake_detector',
      description="package permettant de transformer des sons en np array et plots",
       install_requires=requirements,
      packages=find_packages()) # You can have several packages, try it
