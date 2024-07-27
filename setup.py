import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "MRSI-ToolBox",
    version = "2.0.0",
    description = ("Analysis toolbox for MRSI data"),
    license = "BSD",
    keywords = "example documentation tutorial",
    packages=find_packages(include=['fillgaps','tools','graphplot','filters',
                                    'registration','tractography','connectomics',
                                    'bids','randomize']),
    # long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Beta",
        "Topic :: DataUtils",
        "License :: OSI Approved :: BSD License",
    ],
)
