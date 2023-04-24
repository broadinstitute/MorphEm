from setuptools import setup, find_packages

from my_pip_package import __version__, __author__

setup(
    name='my_pip_package',
    version=__version__,

    url='https://github.com/broadinstitute/MorphEm',
    author=__author__,
    author_email='zitong@broadinstitute.org',

    py_modules=find_packages(),
    install_requires=[
    'faiss-gpu==1.7.2',
    'matplotlib==3.5.3',
    'numpy==1.21.5',
    'pandas==1.5.1',
    'scikit_image==0.19.2',
    'scikit_learn==1.2.2',
    'torch==2.0.0',
    'torchvision==0.15.1',
    'umap'
    ]
    
)