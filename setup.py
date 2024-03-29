'''setup.py
Main setup file for distribution.
'''

__author__ = 'Geoffrey Angus & Richard Diehl Martinez'

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    entry_points={
        'console_scripts': [
            'build_dataset=via.run:build_dataset',
            'build_projection=via.run:build_projection',
            'run_metrics=via.run:run_metrics',
            'enrich_projection=via.run:enrich_projection',
            'run_pagerank=via.run:run_pagerank'
        ]
    },
    name="via",
    version="0.0.1dev",
    author="Geoffrey Angus and Richard Diehl",
    author_email="gangus@stanford.edu",
    description="Research software for the Carta Via project",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/geoffreyangus/Via",
    packages=find_packages(include=['via']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'networkx',
        'click',
        'pandas',
        'pytz',
        'python-dateutil',
        'numpy',
    ],
    python_requires='>=3',
)
