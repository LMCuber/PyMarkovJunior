from setuptools import setup, find_packages


setup(
    name="MarkovJunior",
    version="0.1.0",
    description="Pure-Python port of the MarkovJunior project",
    author="Leo Bozkir",
    author_email="leo.bozkir@outlook.com",
    url="https://github.com/LMCuber/PyMarkovJunior",
    packages=find_packages(),
    install_requires=[
        "pygame", "numpy", "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)