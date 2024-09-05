from setuptools import setup, find_packages

setup(
    name='SyntheticDataAlchemy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'scipy',
        'category_encoders',
    ],
    author='Phillip Peng',
    author_email='ppeng08@gmail.com',
    description='A package for generating synthetic data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/phillip1029/DataAlchemy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)