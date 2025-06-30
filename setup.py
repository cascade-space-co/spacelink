import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spacelink",
    version="0.1.0",
    author="Jacob Portukalian, Arlen Abraham, Brett Gottula",
    author_email="jacob@cascade.space",
    description="Space link budget calculation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cascade-space-co/spacelink",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.0.0,<3.0.0",
        "scipy>=1.10.0,<2.0.0",
        "autopep8>=2.3.2",
        "astropy>=7.0.1,<8.0.0",
        "black>=25.1.0,<26.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Communications",
    ],
    include_package_data=True,
    license="MIT",
)
