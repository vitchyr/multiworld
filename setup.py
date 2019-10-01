from distutils.core import setup
from setuptools import find_packages

setup(
    name='bullet_manipulation',
    packages=find_packages(),
    version='0.0.1',
    description='Bullet manipulation environments',
    long_description=open('./README.md').read(),
    author='Michael Janner',
    author_email='janner@berkeley.edu',
    requires=(),
    zip_safe=True,
    license='MIT'
)