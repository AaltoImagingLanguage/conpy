#! /usr/bin/env python
from setuptools import setup
import codecs
import os

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name='conpy',
        maintainer='Marijn van Vliet',
        maintainer_email='w.m.vanvliet@gmail.com',
        description='Functions and classes for performing connectivity analysis on MEG data.',
        license='BSD-3',
        url='https://github.com/aaltoimaginglanguage/conpy',
        version='1.2',
        download_url='https://github.com/aaltoimaginglanguage/conpy/archive/master.zip',
        long_description=codecs.open('README.md', encoding='utf8').read(),
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        platforms='any',
        packages=['conpy'],
        install_requires=['numpy', 'scipy', 'mne'],
    )
