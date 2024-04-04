#!/usr/bin/env python
from setuptools import setup


setup(name='visinterf',
      description='LBT interferometry at visible wavelength',
      version='0.1',
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url='',
      author_email='lorenzo.busoni@inaf.it',
      author='Lorenzo Busoni',
      license='MIT',
      keywords='adaptive optics',
      packages=['interfoppy',
                'interfoppy.mains',
                ],
      install_requires=["numpy",
                        "astropy",
                        "arte",
                        ],
      test_suite='test',
      package_data={
          'visinterf': ['data/*'],
      },
      include_package_data=True,
      )
