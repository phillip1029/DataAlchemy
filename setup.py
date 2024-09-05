from setuptools import setup, find_packages

setup(
    name='DataAlchemy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'scipy',
        'category_encoders',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for generating synthetic data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/DataAlchemy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)