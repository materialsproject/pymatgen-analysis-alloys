# coding: utf-8

# Original pymatgen add-on template Copyright (c) Materials Virtual Lab
# and distributed under the terms of the Modified BSD License.
# pymatgen-analysis-alloys is Copyright (c) Rachel Woods-Robinson, Matthew Horton

import os

from setuptools import find_namespace_packages, setup

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SETUP_PTH, "README.md")) as readme:
    desc = readme.read()


setup(
    name="pymatgen-analysis-alloys",
    packages=find_namespace_packages(include=["pymatgen.analysis.*"]),
    version="0.0.8",
    install_requires=["pymatgen>=2023.7.17", "shapely>=1.8.2"],
    extras_require={},
    package_data={
        "pymatgen.analysis.alloys": ["*.yaml", "*.json", "*.csv"],
    },
    author="Rachel Woods-Robinson, Matthew Horton",
    author_email="rwoodsrobinson@lbl.gov",
    maintainer="Rachel Woods-Robinson, Matthew Horton",
    url="https://github.com/materialsproject/pymatgen-analysis-alloys",
    description="A pymatgen add-on library with classes useful for describing alloy (disordered) systems.",
    long_description=desc,
    long_description_content_type="text/markdown",
    keywords=["pymatgen"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
