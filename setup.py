from setuptools import setup

setup(
    name='infant',
    version='0.0.1',
    author='Jordi Yaputra',
    packages=['infant'],
    package_dir={'':'src'},
    install_requires=[
        'requests',
        'torch',
        'urllib3',
    ],
)