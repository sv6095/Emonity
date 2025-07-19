from setuptools import setup, find_packages

setup(
    name="emonity",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "librosa",
        "seaborn",
        "matplotlib",
        "scikit-learn",
        "ipython",
        "torch",
        "torchaudio",
        "xgboost",
        "lightgbm",
        "scikit-image",
        "torchvision"
    ],
    author="Your Name",
    description="A Speech Emotion Recognition system using MFCC and Deep Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="<repo-url>",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
