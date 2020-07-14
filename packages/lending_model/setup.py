import io
import os
from pathlib import Path

from setuptools import find_packages, setup


# package meta-data
NAME = 'lending_model'
DESCRIPTION = 'Regression model to predict interest rate of a Lending Club loan'
URL = 'https://github.com/jpbacher/interest-rate-deployment'
EMAIL = 'bacherjp@gmail.com'
AUTHOR = 'Josh Bacher'
REQUIRES_PYTHON = '>=3.7.0'


def list_requirements(fname='requirements.txt'):
    with open(fname) as f:
        return f.read().splitlines()


current_location = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(current_location, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# load package's version module as dictionary
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR/'lending_model'
about = {}
with open(PACKAGE_DIR/'VERSION') as v_file:
    _version = v_file.read().strip()
    about['__version__'] = _version


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'lending_model': ['VERSION']},
    install_requires=list_requirements(),
    extras_require={},
    include_package_data=True,
    license='BSD 3',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
