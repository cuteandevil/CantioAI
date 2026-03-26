"""
CantioAI Setup Script
"""

from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt"""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="cantioai",
    version="0.1.0",
    author="AI Audio Engineer",
    author_email="engineer@cantioai.example",
    description="Hybrid Source-Filter + Neural Vocoder Voice Conversion System",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cantioai",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=21.9b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "isort>=5.9.0",
            "pytest>=6.2.0",
            "pytest-cov>=2.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "tracking": [
            "tensorboard>=2.8.0",
            "wandb>=0.13.0",
        ]
    },
    entry_points={
        'console_scripts': [
            'cantioai-preprocess=scripts.preprocess:main',
            'cantioai-train=scripts.train:main',
            'cantioai-infer=scripts.infer:main',
            'cantioai-evaluate=scripts.evaluate:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)