"""A setuptools based setup module."""
from os import path
from setuptools import setup, find_packages
from io import open

setup(
    name='ml_service',
    include_package_data=True,
    version='0.0.1',
    description='Microservice which is responsible for working with machine learning stuff.',
    long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AlexOmarov/ml_snippets',
    author='Alex Omarov',
    author_email='dungeonswdragons@gmail.com',
    zip_safe=False,
    classifiers=['Programming Language :: Python :: 3.9'],
    keywords='Flask Flask-Assets Machine-learning Tensorflow',
    packages=find_packages(),
    install_requires=[
        'Flask~=2.1.3',
        'Waitress~=2.0.0',
        'TensorFlow-gpu~=2.9.1',
        'Matplotlib~=3.5.2',
        'Numpy~=1.23.2',
        'pyspark~=3.1.1',
        'scikit-learn~=0.24.1',
        'setuptools~=63.4.1',
        'APScheduler~=3.7.0',
        'pandas~=1.4.3',
        'paste~=3.5.0',
        'scipy~=1.7.1'
    ],
    extras_require={
        'dev': [''],
        'test': [''],
        'env': ['']
    },
    entry_points={
        'console_scripts': [
            'install=wsgi:__main__',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/AlexOmarov/ml_service/issues',
        'Source': 'https://github.com/AlexOmarov/ml_service/',
    },
)
