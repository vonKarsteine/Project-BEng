from setuptools import setup

setup(name='gym_uav',
      version='0.0.2',
      packages=['gym_uav', 'gym_uav.envs'],
      install_requires=['gymnasium', 'vtk', 'numpy']
)
