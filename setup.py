from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
   name='uugai_python_color_prediction',
   version='1.0.0',
   description='Color prediction Python library used to find the main colors in an image.', 
   author='uug.ai',
   author_email='support@uug.ai',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   packages=find_packages(),
   install_requires=requirements
)