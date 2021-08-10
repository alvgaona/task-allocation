from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['mrta'],
    package_dir={'': 'src'},
    scripts=['scripts/server_endpoint.py', 'scripts/orientation_publisher.py']
)

setup(**setup_args)
