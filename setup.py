import setuptools

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="KeyDnn",
    version="1.0.0",
    author="keywind",
    author_email="watersprayer127@gmail.com",
    description="Deep Learning Framework Implemented by Kevin Sheu.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keywind127/keydnn",
    project_urls={
        "Bug Tracker": "https://github.com/keywind127/keydnn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=["markdown", "numpy"],
    include_package_data=True,
)
