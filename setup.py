# setup.py
import setuptools

# Use README.md as long description of module
with open("README.md", "r", encoding="utf-8") as readme:
  long_description = readme.read()

setuptools.setup(
  name="ExploreAI-vrnayak",
  version="0.0.1",
  author="Vishal Nayak",
  author_email="vrnayak@umich.edu",
  description="A package of AI algorithms"
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/vrnayak/ExploreAI",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
)