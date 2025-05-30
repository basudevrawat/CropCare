from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="krishibondhu-ai",
    version="1.0.0",
    author="KrishiBondhu AI Team",
    author_email="your.email@example.com",
    description="AI-driven plant disease detection system for Indian farmers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/krishibondhu-ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "krishibondhu=app:main",
        ],
    },
    include_package_data=True,
) 