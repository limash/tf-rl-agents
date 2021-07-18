from setuptools import setup

setup(name='tf_reinforcement_agents',
      version='0.0.1',
      install_requires=[
            'numpy',
            'pandas',
            'tensorflow',
            'tensorflow-probability',
            'tensorflow-addons',
            'matplotlib',
            'kaggle-environments',
            'dm-reverb',
            'ray[default]',
            'gym'
            ]
      )
