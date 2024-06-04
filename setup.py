import setuptools
from setuptools.command.install import install
import os
import subprocess


class CustomInstall(install):
    def run(self):
        install.run(self)

        dir_path = os.path.dirname(os.path.realpath(__file__))




setuptools.setup(
    name='protein_holography_web',
    version='0.1.0',
    author='Gian Marco Visani',
    author_email='gvisan01@.cs.washington.edu',
    description='Learning protein neighborhoods by incorporating rotational symmetry - web version',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/StatPhysBio/protein_holography-web',
    python_requires='>=3.9',
    install_requires='',
    packages=setuptools.find_packages(),
    include_package_data=True,
    cmdclass={"install": CustomInstall}
)
