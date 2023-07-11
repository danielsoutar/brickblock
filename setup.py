from setuptools import setup, find_packages
from os.path import abspath, dirname, join  # HMMM

# Fetches the content from README.md
# This will be used for the "long_description" field
README_MD = open(join(dirname(abspath(__file__)), "README.md")).read()

# Can we not seed this from other files? I'd rather not have to update several
# places...
setup(
    # REQUIRED
    name="sterling",
    # REQUIRED
    version="0.1.0",
    # Not sure?
    packages=find_packages(exclude="tests"),
    # OPTIONAL
    description="A fun visualisation library for those that like boxes",
    # OPTIONAL
    long_description=README_MD,
    # OPTIONAL
    long_description_content_type="text/markdown",
    # OPTIONAL
    url="https://github.com/danielsoutar/sterling",
    # OPTIONAL
    author_name="Daniel Soutar",
    # OPTIONAL
    author_email="danielsoutar144@gmail.com",
    # Classifiers help categorize the project.
    # For a complete list of classifiers, visit https://pypi.org/classifiers
    # OPTIONAL
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3 :: Only"
    ],
    # Keywords are tags that identify your project and help searching for it
    # OPTIONAL
    keywords="sterling, 3D, visualisations",
    # For additional fields, check:
    # https://github.com/pypa/sampleproject/blob/master/setup.py
)
