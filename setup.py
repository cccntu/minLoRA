from setuptools import setup

setup(
    name="minLoRA",
    version="0.0.1",
    author="Jonathan Chang",
    packages=["minlora"],
    description="A PyTorch re-implementation of LoRA",
    license="MIT",
    install_requires=[
        "torch>=1.9.0",
    ],
)
