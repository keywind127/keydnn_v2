import setuptools
from pathlib import Path

ROOT = Path(__file__).parent


def read_requirements(filename: str = "requirements.txt") -> list[str]:
    req_path = ROOT / filename
    if not req_path.exists():
        return []
    reqs: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


long_description = (ROOT / "README_PYPI.md").read_text(encoding="utf-8")

setuptools.setup(
    name="keydnn",
    version="2.0.0a0",  # PEP 440 compliant
    author="keywind",
    author_email="watersprayer127@gmail.com",
    description=(
        "KeyDNN is a lightweight deep learning framework built from scratch "
        "in Python with a strong focus on clean architecture, explicit interfaces, "
        "and a practical CPU/CUDA execution stack."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keywind127/keydnn_v2",
    project_urls={"Bug Tracker": "https://github.com/keywind127/keydnn_v2/issues"},
    license="Apache-2.0",
    license_files=("LICENSE",),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
    ],
    platforms=["Windows"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    include_package_data=True,
    zip_safe=False,
    package_data={
        "keydnn": [
            "infrastructure/native/python/*.dll",
            "infrastructure/native_cuda/keydnn_v2_cuda_native/x64/Release/*.dll",
        ],
    },
)
