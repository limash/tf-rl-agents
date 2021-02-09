from setuptools import setup

setup(name='goose_agent',
      version='0.0.1',
      install_requires=['numpy',
                        'matplotlib',
                        'tensorflow',
                        'ray',
                        'reverb',
                        'gym']
      )
